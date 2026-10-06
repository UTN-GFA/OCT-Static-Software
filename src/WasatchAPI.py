"""Driver for the Awaiba Dragster spectrometer via IMAQ camera interface.

Wraps pylablib's IMAQ interface to control sensor parameters (gain,
offset, integration time, line time) and acquire spectral data.
"""

import time
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from pylablib.devices import IMAQ
from scipy.interpolate import interp1d
from scipy.signal import zoom_fft as scipy_zoom_fft

# Number of dimensions of a single chunk array in IMAQ "chunks" frame format.
CHUNK_NDIM = 3

# Seconds per microsecond, for converting the sensor's line time.
US_TO_S = 1e-6


# Default lower edge of the analysed OPD band, in units of the axial
# resolution 2π/Δk. The DC/autocorrelation lobe is not confined to bins below
# one resolution element: its skirt still exceeds a genuine reflector's peak at
# exactly 1x, so the band starts at a small multiple of it. Measured on
# synthetic reflectors from 30 µm to 4 mm, 1x mislocates every one of them
# while >=1.5x recovers all; 2x keeps a margin.
DC_REJECTION_FACTOR = 2.0


class WasatchAPI:
    """API for controlling the Awaiba Dragster spectrometer.

    Communicates with the spectrometer hardware over IMAQ serial commands
    to configure sensor parameters and acquire raw spectral frames.
    """

    def __init__(self):
        """Initialize WasatchAPI with default sensor parameters."""
        self.camera: IMAQ.IMAQCamera | None = None
        self._camera_id: str | None = None
        self.num_pixels: int = 2048
        self._pixel_array: np.ndarray = np.arange(self.num_pixels)
        self._calibration_constants: np.ndarray = np.array(
            [7.94223e2, 4.62979e-2, -2.60004e-6, -1.48385e-11])
        self._min_opd: float | None = None
        self._max_opd: float | None = None
        self._length_opd: int | None = None
        self._gain: int = 194
        self._offset: int = 255
        self._integration_time: int = 90
        self._line_time: int = 100
        self._camera_on: bool = False

    @property
    def wavelength(self) -> np.ndarray:
        """Calibrated wavelength for each pixel.

        Applies the polynomial calibration constants to the pixel array.

        Returns:
            Wavelength array in nanometers, shape (2048,).

        """
        return np.polynomial.polynomial.polyval(
            self._pixel_array, self._calibration_constants,
        )

    @property
    def min_wavelength(self) -> float:
        """Minimum wavelength in nanometers."""
        return self.wavelength[0]

    @property
    def max_wavelength(self) -> float:
        """Maximum wavelength in nanometers."""
        return self.wavelength[-1]

    @property
    def delta_wavelength(self) -> float:
        """Difference between maximum and minimum wavelength in nanometers."""
        return self.max_wavelength - self.min_wavelength

    @property
    def wavelength_step(self) -> float:
        """Distance between two consecutive wavelengths in nanometers.

        Returns:
            Wavelength step in nm.

        """
        return self.delta_wavelength / (self.num_pixels - 1)

    @property
    def min_wavenumber(self) -> float:
        """Minimum wavenumber in rad/nm corresponding to maximum wavelength."""
        return 2 * np.pi / self.max_wavelength

    @property
    def max_wavenumber(self) -> float:
        """Maximum wavenumber in rad/nm corresponding to minimum wavelength."""
        return 2 * np.pi / self.min_wavelength

    @property
    def delta_wavenumber(self) -> float:
        """Difference between maximum and minimum wavenumber in rad/nm."""
        return self.max_wavenumber - self.min_wavenumber

    @property
    def wavenumber_step(self) -> float:
        """Distance between two consecutive (uniformly spaced) wavenumbers in rad/nm."""
        return self.delta_wavenumber / (self.num_pixels - 1)

    @property
    def axial_resolution(self) -> float:
        """Axial resolution ``OPD_min = 2π / Δk`` in nanometers (teoría).

        The smallest OPD separation the spectral range can resolve. It is the
        physical floor for :attr:`min_opd`, which defaults to a small multiple
        of it to clear the DC/autocorrelation lobe.
        """
        return 2 * np.pi / self.delta_wavenumber

    @property
    def min_opd(self) -> float:
        """Lower bound of the analysed optical path difference (OPD) band, in nm.

        Defaults to ``DC_REJECTION_FACTOR`` times the
        :attr:`axial_resolution` ``2π/Δk`` (teoría). Below that lies the
        DC/autocorrelation term (the non-interferometric background) rather
        than a resolvable reflector, so it is also the frequency at which
        :meth:`zoom_fft` starts evaluating the DFT.

        Assign to this property to skip a near-surface reflection or to
        concentrate the OPD samples on a deeper region of interest; assign
        ``None`` to fall back to the default.
        """
        if self._min_opd is not None:
            return self._min_opd
        return DC_REJECTION_FACTOR * self.axial_resolution

    @min_opd.setter
    def min_opd(self, min_opd_nm: float | None) -> None:
        """Override the lower bound of the analysed OPD band (nanometers)."""
        self._min_opd = min_opd_nm

    @property
    def max_opd(self) -> float:
        """Upper bound of the analysed optical path difference (OPD) band, in nm.

        Defaults to the maximum imaging depth. It is the frequency at which
        :meth:`zoom_fft` stops evaluating the DFT.

        Assign to this property to narrow the band; assign ``None`` to fall
        back to the full imaging depth.
        """
        if self._max_opd is not None:
            return self._max_opd
        return np.pi / self.wavenumber_step

    @max_opd.setter
    def max_opd(self, max_opd_nm: float | None) -> None:
        """Override the upper bound of the analysed OPD band (nanometers)."""
        self._max_opd = max_opd_nm

    @property
    def wavelength_eqspaced(self) -> np.ndarray:
        """Equally spaced wavelength array in nanometers."""
        return np.linspace(self.min_wavelength, self.max_wavelength, self.num_pixels)

    @property
    def wavenumber(self) -> np.ndarray:
        """Wavenumber array in rad/nm, ordered from min to max.

        Derived as ``k = 2π / λ``. Because ``λ`` grows with pixel index, the
        raw array is decreasing; it is flipped so that the returned array is
        monotonically increasing (required by ``scipy.interpolate.interp1d``).
        """
        return np.flip(2 * np.pi / self.wavelength)

    @property
    def wavenumber_eqspaced(self) -> np.ndarray:
        """Equally spaced (uniform) wavenumber array in rad/nm."""
        return np.linspace(self.min_wavenumber, self.max_wavenumber, self.num_pixels)

    @property
    def length_opd(self) -> int:
        """Number of points in the OPD axis. Defaults to ``num_pixels``."""
        if self._length_opd is not None:
            return self._length_opd
        return self.num_pixels

    @length_opd.setter
    def length_opd(self, length_opd: int) -> None:
        """Override the number of OPD samples."""
        self._length_opd = length_opd

    @property
    def opd(self) -> np.ndarray:
        """Optical path difference (OPD) axis in nanometers.

        A uniform ``k`` sampling with step ``dk`` maps, via the DFT, to an OPD
        axis ``opd[m] = 2π·m / (N·dk)``. This property returns the analysed
        band ``[min_opd, max_opd)`` with ``length_opd`` samples, matching bin
        for bin the spectrum produced by :meth:`zoom_fft`.
        """
        return np.linspace(self.min_opd, self.max_opd, self.length_opd,
                           endpoint=False)

    @property
    def camera_id(self) -> str | None:
        """Identifier of the connected camera.

        Returns:
            Camera identifier string, or None if not connected.

        """
        return self._camera_id

    @camera_id.setter
    def camera_id(self, camera_id: str):
        self.select_camera(camera_id)

    @property
    def gain(self) -> int:
        """ADC analog gain setting.

        The Awaiba Dragster sensor uses an inverse ADC gain (analog gain
        for step size of ADC) from -6 to 20 dB. A higher register value
        produces lower gain; firmware may invert this behavior.

        Returns:
            Current gain value (0--255, default 194).

        """
        return self._gain

    @gain.setter
    def gain(self, value: int):
        """Set ADC analog gain.

        Args:
            value: Gain value (0--255).

        """
        _min, _max = 0, 255
        value = max([min([value, _max]), _min])
        self._setter("gain", "gain", value)

    @property
    def offset(self) -> int:
        """ADC black level offset.

        Lower values increase the baseline level; higher values lower it
        closer to black.

        Returns:
            Current offset value (0--255, default 255).

        """
        return self._offset

    @offset.setter
    def offset(self, value: int):
        """Set ADC black level offset.

        Args:
            value: Offset value (0--255).

        """
        _min, _max = 0, 255
        value = max([min([value, _max]), _min])
        self._setter("offset", "offset", value)

    @property
    def integration_time(self) -> int:
        """Sensor integration (collection) time in microseconds.

        The signal is sampled and held for readout after integration.
        Must be at least 2 µs less than the line time.

        Returns:
            Current integration time (0--32767, default 90).

        """
        return self._integration_time

    @integration_time.setter
    def integration_time(self, value: int):
        """Set integration time.

        Args:
            value: Integration time in microseconds (1--32767).

        """
        _min, _max = 1, 32767
        value = max([min([value, _max]), _min])

        if value > self.line_time:
            self.line_time = value + 2

        self._setter("int", "integration_time", value)

    @property
    def line_time(self) -> int:
        """Line period including transfer/reset overhead.

        Determines the maximum achievable line rate in microseconds.

        Returns:
            Current line time (22--65535, default 100).

        """
        return self._line_time

    @line_time.setter
    def line_time(self, value: int):
        """Set line time.

        Args:
            value: Line time value in microseconds (22--65535).

        """
        _min, _max = 22, 65535
        value = max([min([value, _max]), _min])

        if value < self.integration_time:
            self.integration_time = value - 2

        self._setter("ltm", "line_time", value)

    @property
    def camera_on(self) -> int:
        """Whether the camera sensor is currently enabled (1) or not (0)."""
        return self._camera_on

    @camera_on.setter
    def camera_on(self, camera_on: bool) -> None:
        """Enable (True) or disable (False) the sensor via the ``lsc`` command."""
        lsc: int = 1 if camera_on else 0
        self._setter("lsc", "camera_on", lsc)

    def _setter(self, key: str, parameter: str, value: int):
        """Send parameter command to hardware and update local cache.

        Args:
            key: Hardware command key (e.g. 'gain', 'int', 'ltm').
            parameter: Internal attribute name without leading underscore.
            value: Value to set.

        Raises:
            ValueError: If the hardware command fails.

        """
        try:
            self.set_parameter(key, value)
            self.__setattr__(f"_{parameter}", value)
        except Exception as e:
            raise ValueError(f"Error setting parameter: {e}")

    def list_cameras(self):
        """List available IMAQ cameras.

        Returns:
            List of camera identifiers.

        """
        return IMAQ.list_cameras()

    def message_camera(self, message: str):
        """Send a serial message to the camera and print the response.

        Args:
            message: Raw serial command string.

        Raises:
            ValueError: If no camera is selected.

        """
        if self.camera is None:
            raise ValueError("No camera selected")

        self.camera.serial_write(message)
        print(f"API: {message}\n Wasatch: {self.camera.serial_readline()}")

    def set_parameter(self, parameter: str, value: int):
        """Set a hardware parameter via serial command.

        Args:
            parameter: Parameter command key.
            value: Integer value to set.

        Raises:
            ValueError: If no camera is selected.

        """
        if self.camera is None:
            raise ValueError("No camera selected")

        self.message_camera(f"{parameter} {value}\r")

    def select_camera(self, camera_id: str):
        """Open a camera connection and apply stored settings.

        Closes any previously open camera before connecting. Sends
        ``init`` command and pushes current gain, offset, integration
        time, and line time to the hardware.

        Args:
            camera_id: IMAQ camera identifier string.

        Raises:
            ValueError: If the camera cannot be opened.

        """
        if self.camera is not None:
            self.camera.close()

        try:
            self.camera = IMAQ.IMAQCamera(camera_id)
            self._camera_id = camera_id
            self.message_camera("init\r")
            print(f"Using camera {camera_id}")
            self.gain = self._gain
            self.offset = self._offset
            self.integration_time = self._integration_time
            self.line_time = self._line_time
            self.camera_on = True
        except Exception as e:
            raise ValueError(f"Error selecting camera: {e}")

    def close_camera(self):
        """Close the active camera connection.

        Raises:
            ValueError: If no camera is selected.

        """
        if self.camera is None:
            raise ValueError("No camera selected")

        self.camera.close()
        self._camera_id = None

    def snap(self) -> np.ndarray:
        """Acquire a single frame from the camera.

        A frame is a block of consecutive sensor lines, each one a full
        spectrum, acquired one every :attr:`line_time` microseconds. Returns a
        zero-filled frame when no camera is selected (so callers can run
        without hardware for testing).

        Returns:
            2-D array of irradiance values, shape ``(nlines, num_pixels)``.

        """
        if self.camera is None:
            return np.zeros((1, self.num_pixels))

        if not self.camera_on:
            self.camera_on = True

        return np.array(self.camera.snap())

    def acquire(self,
                acquisition_time: int,
                save_data: bool = True,
                filename: str | None = None) -> tuple[np.ndarray, float]:
        """Acquire multiple frames over a given duration.

        Calculates the number of frames from the acquisition time and
        the current line time, then performs a sequence acquisition.
        Records a timestamp for each frame using ``time.perf_counter()``.

        Args:
            acquisition_time: Total acquisition time in microseconds.
                Converted to a frame count as
                ``nframes = 1 + acquisition_time / (100 * line_time)``.
            save_data: If True, save the acquired signal to
                ``data/<filename>.npy`` (default timestamped name).
            filename: Base name for the saved file (without extension).
                Ignored when ``save_data`` is False.

        Returns:
            Tuple of (signal, measured_time) where:
                - frames: 2-D array of shape (nframes, 2048).
                - measured_time: measured wall-clock time in milliseconds.

        Raises:
            ValueError: If no camera is selected.

        """
        if self.camera is None:
            raise ValueError("No camera selected")

        # acquisition_time = frames * frame_time = frames * 100 * line_time
        nframes = 1 + int(acquisition_time / (100 * self.line_time))

        if not self.camera_on:
            self.camera_on = True

        self.camera.setup_acquisition(mode="sequence", nframes=nframes)
        self.camera.set_frame_format("chunks")
        self.camera.start_acquisition()

        _start_time = time.perf_counter()
        self.camera.wait_for_frame(since="lastread", nframes=nframes,
                                   timeout=(None, 15.0))  # type: ignore
        _stop_time = time.perf_counter()

        rng = self.camera.get_new_images_range()
        if rng is None:
            frames = np.empty((0, self.num_pixels))

        chunks = self.camera.read_multiple_images(
            rng=rng, missing_frame="skip",
        )
        # chunks may be a list of 3-D arrays (chunks mode) or a list of
        # 2-D frames (fallback); stack them all into (n_read, 2048).
        if len(chunks) and chunks[0].ndim == CHUNK_NDIM:
            frames = np.concatenate(chunks, axis=0)
        else:
            frames = np.vstack(chunks)

        self.camera.stop_acquisition()
        self.camera.clear_acquisition()

        measured_time_ms = (_stop_time - _start_time) * 1e3

        if save_data:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"acquisition_{timestamp}"

            filepath = Path(f"data/{filename}")
            np.savez(filepath, frames=frames,
                     acquisition_time=measured_time_ms)

        return frames, measured_time_ms

    def zoom_fft(self, signal: ArrayLike) -> np.ndarray:
        """Compute the optical path difference (OPD) spectrum from irradiance.

        Resamples the interferogram onto a uniform wavenumber grid and applies
        ``scipy.signal.zoom_fft`` to extract the OPD spectrum over the analysed
        band ``[min_opd, max_opd)``.

        The interferometric signal ``I(k) = 2·A_s·A_r·cos(OPD·k + φ)`` is a
        function of the wavenumber ``k`` (rad/nm); its OPD is the spatial
        frequency of that oscillation (teoría, "Dominio de Fourier"). A uniform
        ``k`` sampling with step ``dk`` maps, via the DFT, to an OPD axis
        ``opd[m] = 2π·m / (N·dk)``. ``scipy.signal.zoom_fft`` evaluates the DFT
        over an arbitrary frequency band directly, so the transform is
        restricted to the band delimited by :attr:`min_opd` and
        :attr:`max_opd`. Narrowing that band concentrates the available
        ``length_opd`` samples on the region of interest, increasing the OPD
        sampling density there.

        ``endpoint=False`` mirrors the ``opd`` axis, so bin ``m`` of the
        returned spectrum corresponds exactly to ``opd[m]``.

        Args:
            signal: Measured irradiance ordered by pixel index (matching
                ``self.wavelength``). Either a single line of shape
                ``(num_pixels,)`` or a frame of shape ``(nlines, num_pixels)``
                as returned by :meth:`snap`.

        Returns:
            Absolute OPD spectrum aligned with the ``opd`` axis: shape
            ``(length_opd,)`` for a single line, ``(nlines, length_opd)`` for a
            frame.

        """
        # ``wavenumber`` is flipped to be monotonically increasing (required by
        # interp1d), so the irradiance must be flipped alongside it to keep each
        # sample paired with the k it was measured at. Flip only the pixel axis:
        # a bare np.flip would also reverse the line order of a 2-D frame.
        signal_eqspaced = interp1d(
            self.wavenumber, np.flip(np.asarray(signal), axis=-1),
        )(self.wavenumber_eqspaced)

        # k is in rad/nm, so the sampling rate of the interferogram is
        # 1/dk cycles per (unit k). OPD (nm) = 2π·(frequency in cycles/nm).
        # interp1d and zoom_fft both act on the last axis, so a whole frame is
        # transformed in one vectorized call.
        fs = 1.0 / self.wavenumber_step
        spectrum = np.abs(
            scipy_zoom_fft(
                signal_eqspaced,
                [self.min_opd / (2 * np.pi), self.max_opd / (2 * np.pi)],
                self.length_opd,
                fs=fs,  # type: ignore
                endpoint=False,
            )
        )

        return spectrum

    def peak_opd(self, spectrum: ArrayLike) -> tuple[float, float]:
        """Locate the dominant optical path difference (OPD) in one spectrum.

        The interference term ``I(k) = 2·A_s·A_r·cos(OPD·k + φ)`` transforms
        into a delta at ``z = OPD`` (teoría, "Dominio de Fourier"), so the
        sample position of the largest peak is the measured OPD.

        The search covers the whole analysed band, which :meth:`zoom_fft`
        already restricts to ``[min_opd, max_opd)``. Since ``min_opd`` defaults
        to a multiple of the axial resolution, the DC/autocorrelation term is
        excluded by construction; raise ``min_opd`` further to skip a
        near-surface reflection.

        Use :meth:`peak_opds` for a whole frame.

        Args:
            spectrum: Absolute OPD spectrum of a single line, as returned by
                :meth:`zoom_fft`, aligned with the :attr:`opd` axis.

        Returns:
            Tuple of (peak_opd, peak_amplitude) where:
                - peak_opd: OPD of the dominant peak in nanometers, or
                  ``float('nan')`` if the spectrum is empty.
                - peak_amplitude: spectral amplitude at that peak.

        """
        spectrum = np.asarray(spectrum)

        if spectrum.size == 0:
            return float("nan"), float("nan")

        index = int(np.argmax(spectrum))
        return float(self.opd[index]), float(spectrum[index])

    def peak_opds(self, spectra: ArrayLike) -> np.ndarray:
        """Locate the dominant OPD of every line in a frame.

        Vectorized counterpart of :meth:`peak_opd`: takes the per-line argmax
        in one pass, which is what makes per-line peak tracking affordable at
        line rates of tens of kHz.

        Args:
            spectra: Absolute OPD spectra of shape ``(nlines, length_opd)``,
                as returned by :meth:`zoom_fft` on a frame. A single 1-D
                spectrum is accepted and treated as one line.

        Returns:
            Peak OPD per line in nanometers, shape ``(nlines,)``.

        """
        spectra = np.atleast_2d(np.asarray(spectra))

        if spectra.size == 0:
            return np.empty(0)

        return self.opd[np.argmax(spectra, axis=-1)]

    def line_timestamps(self, nlines: int, frame_end_s: float) -> np.ndarray:
        """Reconstruct the acquisition instant of each line in a frame.

        The sensor reads one line every :attr:`line_time` microseconds, so
        within a frame the lines are equally spaced in time. Only the frame as
        a whole carries a host timestamp, so the individual line times are
        derived backwards from the end of the frame: the last line is the most
        recent one, and line ``i`` precedes it by ``(nlines - 1 - i)``
        line periods.

        Args:
            nlines: Number of lines in the frame.
            frame_end_s: Host timestamp of the end of the frame, in seconds
                (``time.perf_counter()``).

        Returns:
            Acquisition time of each line in seconds, shape ``(nlines,)``,
            increasing and ending at ``frame_end_s``.

        """
        line_period_s = self.line_time * US_TO_S
        return frame_end_s - line_period_s * np.arange(nlines - 1, -1, -1)
