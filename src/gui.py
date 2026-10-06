"""Minimal real-time user interface for the OCT spectrometer.

Displays three synchronized live plots:

1. Raw irradiance versus wavelength (nm) --- the measured interferogram.
2. Fourier-domain spectrum versus optical path difference (µm) --- the
   ``zoom_fft`` output, where each reflector appears as a peak.
3. Dominant OPD versus time over a rolling 5-second window.

Acquisition and Fourier processing run in a worker thread so the event loop
stays responsive at high line rates; the widgets are only touched from the GUI
thread through Qt signals.
"""

import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from src.WasatchAPI import WasatchAPI

# Rolling time span of the OPD-versus-time plot, in seconds.
TIME_WINDOW_S = 5.0

# Nanometers per micrometer, used to display OPD axes in µm.
NM_PER_UM = 1e3

# Target refresh period of the acquisition loop, in milliseconds.
REFRESH_MS = 30


class AcquisitionWorker(QtCore.QObject):
    """Acquires frames and computes their OPD spectrum off the GUI thread.

    Runs a self-rescheduling loop: each iteration snaps one frame, transforms
    every line with :meth:`WasatchAPI.zoom_fft`, locates the dominant OPD of
    each line and emits everything as a single signal.

    Attributes:
        frameReady: Emitted per frame with
            ``(raw_line, spectrum, peak_opds_nm, line_times_s)``, where the
            first two describe the frame's last line (the most recent one) and
            the last two carry every line in the frame.
        errorRaised: Emitted with a message when acquisition fails.

    """

    frameReady = QtCore.Signal(object, object, object, object)
    errorRaised = QtCore.Signal(str)

    def __init__(self, api: WasatchAPI):
        """Store the API handle and prepare the worker in a stopped state.

        Args:
            api: Spectrometer API used to acquire and transform frames.

        """
        super().__init__()
        self.api = api
        self._running = False
        self._timer: QtCore.QTimer | None = None

    @QtCore.Slot()
    def start(self) -> None:
        """Create the loop timer inside the worker thread and start it."""
        if self._running:
            return

        self._running = True
        self._timer = QtCore.QTimer()
        self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._acquire_once)
        self._timer.start(REFRESH_MS)

    @QtCore.Slot()
    def stop(self) -> None:
        """Stop the loop timer and mark the worker as idle."""
        self._running = False

        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    @QtCore.Slot()
    def _acquire_once(self) -> None:
        """Acquire, transform and publish a single frame."""
        if not self._running:
            return

        try:
            frame = np.atleast_2d(self.api.snap())
            # zoom_fft is vectorized over the last axis, so the whole frame is
            # transformed at once rather than line by line.
            spectra = self.api.zoom_fft(frame)
            peak_opds = self.api.peak_opds(spectra)
            frame_end = time.perf_counter()
            line_times = self.api.line_timestamps(len(frame), frame_end)
        except Exception as error:  # noqa: BLE001 - surfaced in the GUI
            self.stop()
            self.errorRaised.emit(str(error))
            return

        # Plots 1 and 2 show the most recent line; plot 3 gets every line.
        self.frameReady.emit(frame[-1], spectra[-1], peak_opds, line_times)


class MainWindow(QtWidgets.QMainWindow):
    """Main window holding the control panel and the three live plots."""

    def __init__(self, api: WasatchAPI):
        """Build the interface and wire it to the acquisition worker.

        Args:
            api: Spectrometer API instance to control and visualize.

        """
        super().__init__()
        self.api = api
        self._peak_times: deque[float] = deque()
        self._peak_values: deque[float] = deque()
        self._frame_count = 0
        self._line_count = 0
        self._fps_reference = time.perf_counter()

        self.setWindowTitle("High-Frequency OCT")
        self.resize(1180, 820)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self._build_controls(), 0)
        layout.addWidget(self._build_plots(), 1)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Detenido")
        self._start_worker()

    def _build_controls(self) -> QtWidgets.QWidget:
        """Create the left-hand control panel.

        Returns:
            Panel widget with camera selection, sensor parameters and the
            acquisition start/stop button.

        """
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(260)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(6)

        self.camera_box = QtWidgets.QComboBox()
        self.camera_box.addItems([str(name)
                                 for name in self.api.list_cameras()])
        connect_button = QtWidgets.QPushButton("Conectar cámara")
        connect_button.clicked.connect(self._connect_camera)

        layout.addWidget(QtWidgets.QLabel("Cámara"))
        layout.addWidget(self.camera_box)
        layout.addWidget(connect_button)
        layout.addSpacing(10)

        self.gain_spin = self._add_parameter(
            layout, "Ganancia", (0, 255), self.api.gain, "gain",
        )
        self.offset_spin = self._add_parameter(
            layout, "Offset", (0, 255), self.api.offset, "offset",
        )
        self.integration_spin = self._add_parameter(
            layout, "T. integración (µs)", (1, 32767),
            self.api.integration_time, "integration_time",
        )
        self.line_spin = self._add_parameter(
            layout, "T. línea (µs)", (22,
                                      65535), self.api.line_time, "line_time",
        )

        layout.addSpacing(10)
        self._build_band_controls(layout)

        self.autoscale_check = QtWidgets.QCheckBox("Autoescala vertical")
        self.autoscale_check.setChecked(True)
        layout.addWidget(self.autoscale_check)

        layout.addSpacing(10)
        self.run_button = QtWidgets.QPushButton("Iniciar")
        self.run_button.setCheckable(True)
        self.run_button.toggled.connect(self._toggle_acquisition)
        layout.addWidget(self.run_button)

        layout.addSpacing(10)
        self.info_label = QtWidgets.QLabel()
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        self._refresh_info_label()
        layout.addStretch(1)

        return panel

    def _build_band_controls(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Add the OPD analysis-band controls to the panel.

        These bound the band ``zoom_fft`` actually evaluates, not merely the
        view: narrowing it concentrates the ``length_opd`` samples on the
        region of interest, improving the OPD sampling density there.

        Args:
            layout: Layout the widgets are appended to.

        """
        layout.addWidget(QtWidgets.QLabel("Banda de análisis OPD (µm)"))

        self.min_opd_spin = QtWidgets.QDoubleSpinBox()
        self.min_opd_spin.setDecimals(2)
        self.min_opd_spin.setSingleStep(10.0)
        self.min_opd_spin.setKeyboardTracking(False)
        self.min_opd_spin.setPrefix("mín  ")
        layout.addWidget(self.min_opd_spin)

        self.max_opd_spin = QtWidgets.QDoubleSpinBox()
        self.max_opd_spin.setDecimals(2)
        self.max_opd_spin.setSingleStep(50.0)
        self.max_opd_spin.setKeyboardTracking(False)
        self.max_opd_spin.setPrefix("máx  ")
        layout.addWidget(self.max_opd_spin)

        self._refresh_opd_spin_limits()
        self.min_opd_spin.valueChanged.connect(self._apply_opd_band)
        self.max_opd_spin.valueChanged.connect(self._apply_opd_band)

        reset_button = QtWidgets.QPushButton("Banda completa")
        reset_button.clicked.connect(self._reset_opd_band)
        layout.addWidget(reset_button)

    def _refresh_opd_spin_limits(self) -> None:
        """Sync the band spin boxes with the API without re-emitting signals.

        The lower spin box is floored at the axial resolution ``2π/Δk``: no
        narrower band is physically meaningful. Each spin box is bounded by the
        other so the band cannot invert.
        """
        floor_um = self.api.axial_resolution / NM_PER_UM
        ceiling_um = self.api.max_opd * 2 / NM_PER_UM
        minimum_um = self.api.min_opd / NM_PER_UM
        maximum_um = self.api.max_opd / NM_PER_UM

        for spin in (self.min_opd_spin, self.max_opd_spin):
            spin.blockSignals(True)

        self.min_opd_spin.setRange(floor_um, maximum_um - floor_um)
        self.min_opd_spin.setValue(minimum_um)
        self.max_opd_spin.setRange(minimum_um + floor_um, ceiling_um)
        self.max_opd_spin.setValue(maximum_um)

        for spin in (self.min_opd_spin, self.max_opd_spin):
            spin.blockSignals(False)

    def _refresh_info_label(self) -> None:
        """Show the active band and the resulting OPD sampling density."""
        step_nm = float(self.api.opd[1] - self.api.opd[0])
        self.info_label.setText(
            f"Banda: {self.api.min_opd / NM_PER_UM:.2f}–"
            f"{self.api.max_opd / NM_PER_UM:.2f} µm\n"
            f"Muestreo: {step_nm:.1f} nm/bin ({self.api.length_opd} bins)\n"
            f"Resolución axial: {self.api.axial_resolution / NM_PER_UM:.2f} µm\n"
            f"λ: {self.api.min_wavelength:.1f}–{self.api.max_wavelength:.1f} nm",
        )

    def _add_parameter(self,
                       layout: QtWidgets.QVBoxLayout,
                       label: str,
                       limits: tuple[int, int],
                       value: int,
                       attribute: str) -> QtWidgets.QSpinBox:
        """Add a labelled spin box bound to an API property.

        Args:
            layout: Layout the widgets are appended to.
            label: Text shown above the spin box.
            limits: Tuple of (minimum, maximum) accepted values.
            value: Initial value.
            attribute: Name of the :class:`WasatchAPI` property to write.

        Returns:
            The created spin box.

        """
        minimum, maximum = limits
        spin = QtWidgets.QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        spin.valueChanged.connect(
            lambda new_value: self._apply_parameter(attribute, new_value),
        )

        layout.addWidget(QtWidgets.QLabel(label))
        layout.addWidget(spin)

        return spin

    def _build_plots(self) -> QtWidgets.QWidget:
        """Create the three stacked live plots.

        Returns:
            Graphics layout widget containing the raw, Fourier and
            OPD-versus-time plots.

        """
        pg.setConfigOptions(antialias=True, background="k", foreground="w")
        graphics = pg.GraphicsLayoutWidget()

        raw_plot = graphics.addPlot(row=0, col=0, title="Señal cruda")
        raw_plot.setLabel("bottom", "Longitud de onda", units="nm")
        raw_plot.setLabel("left", "Irradiancia", units="cuentas")
        raw_plot.showGrid(x=True, y=True, alpha=0.2)
        self.raw_curve = raw_plot.plot(pen=pg.mkPen("#4fc3f7", width=1))
        self.raw_plot = raw_plot

        fft_plot = graphics.addPlot(row=1, col=0, title="Transformada (OPD)")
        fft_plot.setLabel("bottom", "Diferencia de camino óptico", units="µm")
        fft_plot.setLabel("left", "Amplitud")
        fft_plot.showGrid(x=True, y=True, alpha=0.2)
        self.fft_curve = fft_plot.plot(pen=pg.mkPen("#81c784", width=1))
        self.peak_marker = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(
                "#ff8a65", width=1, style=QtCore.Qt.PenStyle.DashLine,
            ),
        )
        fft_plot.addItem(self.peak_marker)
        self.fft_plot = fft_plot

        peak_plot = graphics.addPlot(
            row=2, col=0, title="Pico OPD por línea vs tiempo (5 s)")
        peak_plot.setLabel("bottom", "Tiempo", units="s")
        peak_plot.setLabel("left", "OPD del pico", units="µm")
        peak_plot.showGrid(x=True, y=True, alpha=0.2)
        peak_plot.setXRange(-TIME_WINDOW_S, 0.0)
        # One point per line, not per frame: thousands of samples accumulate
        # over the window, so they are drawn as a scatter instead of a polyline
        # that would smear them together.
        self.peak_curve = peak_plot.plot(
            pen=None, symbol="o", symbolSize=2,
            symbolPen=None, symbolBrush="#ffd54f",
        )
        self.peak_plot = peak_plot

        # The OPD axis follows the analysis band, so it is rebuilt whenever the
        # band changes rather than cached once.
        self._wavelength_axis = self.api.wavelength
        self._opd_axis_um = self.api.opd / NM_PER_UM
        fft_plot.setXRange(self._opd_axis_um[0], self._opd_axis_um[-1])

        return graphics

    @QtCore.Slot()
    def _apply_opd_band(self) -> None:
        """Push the selected analysis band to the API and rescale the plot.

        Writing :attr:`WasatchAPI.min_opd` / :attr:`WasatchAPI.max_opd` changes
        the band ``zoom_fft`` evaluates, so the OPD axis is rebuilt and the
        rolling peak history cleared: earlier samples came from a different
        band and are no longer comparable.
        """
        minimum_um = self.min_opd_spin.value()
        maximum_um = self.max_opd_spin.value()

        if maximum_um - minimum_um < self.api.axial_resolution / NM_PER_UM:
            return

        self.api.min_opd = minimum_um * NM_PER_UM
        self.api.max_opd = maximum_um * NM_PER_UM
        self._rebuild_opd_axis()

    @QtCore.Slot()
    def _reset_opd_band(self) -> None:
        """Restore the full analysis band and refresh the controls."""
        self.api.min_opd = None
        self.api.max_opd = None
        self._refresh_opd_spin_limits()
        self._rebuild_opd_axis()

    def _rebuild_opd_axis(self) -> None:
        """Recompute the cached OPD axis after the band changed."""
        self._opd_axis_um = self.api.opd / NM_PER_UM
        self.fft_plot.setXRange(self._opd_axis_um[0], self._opd_axis_um[-1])

        # Bounds are interdependent, so keep each spin box's range in step.
        floor_um = self.api.axial_resolution / NM_PER_UM
        self.min_opd_spin.blockSignals(True)
        self.max_opd_spin.blockSignals(True)
        self.min_opd_spin.setMaximum(self.max_opd_spin.value() - floor_um)
        self.max_opd_spin.setMinimum(self.min_opd_spin.value() + floor_um)
        self.min_opd_spin.blockSignals(False)
        self.max_opd_spin.blockSignals(False)

        self._peak_times.clear()
        self._peak_values.clear()
        self.peak_curve.setData([], [])
        self._refresh_info_label()

    def _start_worker(self) -> None:
        """Move the acquisition worker into its own thread and connect it."""
        self.thread = QtCore.QThread()
        self.worker = AcquisitionWorker(self.api)
        self.worker.moveToThread(self.thread)
        self.worker.frameReady.connect(self._update_plots)
        self.worker.errorRaised.connect(self._report_error)
        self.thread.start()

    @QtCore.Slot(bool)
    def _toggle_acquisition(self, running: bool) -> None:
        """Start or stop the acquisition loop.

        Args:
            running: True when the button is pressed in (acquire), False to
                stop.

        """
        self.run_button.setText("Detener" if running else "Iniciar")

        if running:
            self._peak_times.clear()
            self._peak_values.clear()
            self._frame_count = 0
            self._line_count = 0
            self._fps_reference = time.perf_counter()
            QtCore.QMetaObject.invokeMethod(
                self.worker, "start", QtCore.Qt.ConnectionType.QueuedConnection,
            )
            self.statusBar().showMessage("Adquiriendo…")
        else:
            QtCore.QMetaObject.invokeMethod(
                self.worker, "stop", QtCore.Qt.ConnectionType.QueuedConnection,
            )
            self.statusBar().showMessage("Detenido")

    @QtCore.Slot(object, object, object, object)
    def _update_plots(self,
                      raw_line: np.ndarray,
                      spectrum: np.ndarray,
                      peak_opds: np.ndarray,
                      line_times: np.ndarray) -> None:
        """Redraw the three plots with a newly acquired frame.

        Plots 1 and 2 show the frame's most recent line; plot 3 accumulates the
        peak of *every* line, each at its own acquisition time, so it resolves
        motion happening within a frame.

        Args:
            raw_line: Irradiance per pixel of the last line, shape
                ``(num_pixels,)``.
            spectrum: Absolute OPD spectrum of that line.
            peak_opds: Peak OPD of every line in the frame, in nanometers.
            line_times: ``time.perf_counter()`` instant of each line, seconds.

        """
        self.raw_curve.setData(self._wavelength_axis, raw_line)

        # A frame computed just before a band change carries the previous axis
        # length; drop it rather than pairing mismatched arrays.
        if len(spectrum) != len(self._opd_axis_um):
            return

        self.fft_curve.setData(self._opd_axis_um, spectrum)

        if self.autoscale_check.isChecked():
            self._autoscale_spectrum(spectrum)

        self._append_peaks(peak_opds, line_times)

    def _append_peaks(self,
                      peak_opds: np.ndarray,
                      line_times: np.ndarray) -> None:
        """Add one point per line to the rolling OPD-versus-time trace.

        Args:
            peak_opds: Peak OPD of every line in the frame, in nanometers.
            line_times: Acquisition instant of each line, in seconds.

        """
        now = float(line_times[-1])

        valid = np.isfinite(peak_opds)
        self._peak_times.extend(line_times[valid])
        self._peak_values.extend(peak_opds[valid] / NM_PER_UM)

        if not self._peak_times:
            return

        # Drop samples that fell out of the rolling window.
        while self._peak_times and now - self._peak_times[0] > TIME_WINDOW_S:
            self._peak_times.popleft()
            self._peak_values.popleft()

        values = np.array(self._peak_values)
        self.peak_curve.setData(np.array(self._peak_times) - now, values)
        self.peak_marker.setPos(values[-1])
        self._update_status(values[-1] * NM_PER_UM, now, len(peak_opds))

    def _autoscale_spectrum(self, spectrum: np.ndarray) -> None:
        """Scale the Fourier plot to the peaks in the analysed band.

        ``zoom_fft`` already restricts the spectrum to ``[min_opd, max_opd)``,
        which excludes the DC/autocorrelation term, so the whole spectrum can
        drive the scaling.

        Args:
            spectrum: Absolute OPD spectrum aligned with the OPD axis.

        """
        if spectrum.size == 0:
            return

        maximum = float(spectrum.max())
        self.fft_plot.setYRange(0.0, maximum * 1.1 if maximum > 0 else 1.0)

    def _update_status(self, peak_opd: float, timestamp: float,
                       nlines: int = 1) -> None:
        """Refresh the status bar with the peak OPD and the acquisition rate.

        Args:
            peak_opd: Most recent dominant OPD in nanometers.
            timestamp: ``time.perf_counter()`` value of the acquisition.
            nlines: Lines carried by this frame, counted towards the line rate.

        """
        self._frame_count += 1
        self._line_count += nlines
        elapsed = timestamp - self._fps_reference

        if elapsed < 1.0:
            return

        fps = self._frame_count / elapsed
        lines_per_second = self._line_count / elapsed
        self._frame_count = 0
        self._line_count = 0
        self._fps_reference = timestamp
        self.statusBar().showMessage(
            f"Pico OPD: {peak_opd / NM_PER_UM:.2f} µm    |    "
            f"{fps:.1f} fps    |    {lines_per_second:.0f} líneas/s",
        )

    def _apply_parameter(self, attribute: str, value: int) -> None:
        """Write a sensor parameter to the hardware.

        Args:
            attribute: Name of the :class:`WasatchAPI` property to set.
            value: New value.

        """
        try:
            setattr(self.api, attribute, value)
        except Exception as error:  # noqa: BLE001 - surfaced in the GUI
            self._report_error(str(error))

    def _connect_camera(self) -> None:
        """Open the camera selected in the combo box."""
        camera_id = self.camera_box.currentText()

        if not camera_id:
            self._report_error("No hay cámaras disponibles")
            return

        try:
            self.api.select_camera(camera_id)
        except Exception as error:  # noqa: BLE001 - surfaced in the GUI
            self._report_error(str(error))
            return

        self.gain_spin.setValue(self.api.gain)
        self.offset_spin.setValue(self.api.offset)
        self.integration_spin.setValue(self.api.integration_time)
        self.line_spin.setValue(self.api.line_time)
        self.statusBar().showMessage(f"Conectado a {camera_id}")

    @QtCore.Slot(str)
    def _report_error(self, message: str) -> None:
        """Show an error in the status bar and release the run button.

        Args:
            message: Text to display.

        """
        self.run_button.setChecked(False)
        self.statusBar().showMessage(f"Error: {message}")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Stop acquisition and join the worker thread before closing.

        Args:
            event: Qt close event.

        """
        self.worker.stop()
        self.thread.quit()
        self.thread.wait(2000)
        super().closeEvent(event)


def run(api: WasatchAPI | None = None) -> int:
    """Launch the user interface.

    Args:
        api: Spectrometer API to drive. Defaults to a fresh
            :class:`WasatchAPI`, which returns zero-filled frames until a
            camera is connected.

    Returns:
        Qt application exit code.

    """
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(api if api is not None else WasatchAPI())
    window.show()

    return application.exec()
