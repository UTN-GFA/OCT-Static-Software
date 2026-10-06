"""Hardware-free stand-in for :class:`~src.WasatchAPI.WasatchAPI`.

Generates synthetic interferograms so the user interface and the Fourier
processing chain can be exercised without a spectrometer attached. The
simulated sample mirror oscillates sinusoidally, reproducing the
``OPD(t) = x_s - x_r - β·sin(Ω·t)`` modulation described in the theory notes
("Modulación de la señal"), which makes the OPD-versus-time plot show a clean,
predictable oscillation.
"""

import time

import numpy as np
from numpy.typing import ArrayLike

from src.WasatchAPI import US_TO_S, WasatchAPI


class SimulatedWasatchAPI(WasatchAPI):
    """Drop-in replacement for :class:`WasatchAPI` that fabricates spectra.

    Overrides only :meth:`snap`; every wavelength, wavenumber and OPD property
    is inherited, so the simulated data lives on exactly the same axes as real
    acquisitions.

    Attributes:
        mean_opd: Static optical path difference in nanometers, i.e. the
            ``x_s - x_r`` offset of the sample arm.
        modulation_amplitude: Peak excursion ``β`` of the oscillating mirror
            in nanometers.
        modulation_frequency: Oscillation frequency ``Ω / 2π`` in hertz.
        noise_level: Standard deviation of the additive Gaussian noise,
            relative to the fringe amplitude.

    """

    def __init__(self,
                 mean_opd: float = 3.0e5,
                 modulation_amplitude: float = 5.0e4,
                 modulation_frequency: float = 0.4,
                 noise_level: float = 0.05,
                 nlines: int = 100):
        """Initialize the simulator with a sinusoidally moving sample mirror.

        Args:
            mean_opd: Static OPD in nanometers (default 300 µm).
            modulation_amplitude: Mirror excursion in nanometers (default 50 µm).
            modulation_frequency: Mirror frequency in hertz.
            noise_level: Relative standard deviation of the additive noise.
            nlines: Lines per frame, matching the real camera's block readout.

        """
        super().__init__()
        self.mean_opd = mean_opd
        self.modulation_amplitude = modulation_amplitude
        self.modulation_frequency = modulation_frequency
        self.noise_level = noise_level
        self.nlines = nlines
        self._start_time = time.perf_counter()
        self._rng = np.random.default_rng()

    def opd_at(self, elapsed: ArrayLike) -> np.ndarray:
        """Mirror OPD at the given times since start-up.

        Implements ``OPD(t) = x_s - x_r - β·sin(Ω·t)`` (teoría, "Modulación de
        la señal").

        Args:
            elapsed: Times since start-up in seconds.

        Returns:
            OPD in nanometers, same shape as ``elapsed``.

        """
        return self.mean_opd - self.modulation_amplitude * np.sin(
            2 * np.pi * self.modulation_frequency * np.asarray(elapsed),
        )

    @property
    def current_opd(self) -> float:
        """Instantaneous OPD of the simulated sample arm in nanometers."""
        return float(self.opd_at(time.perf_counter() - self._start_time))

    def snap(self) -> np.ndarray:
        """Generate one synthetic frame of consecutive lines.

        Builds ``I(k) = S(k)·[1 + V·cos(OPD·k)]`` on the true (non-uniform)
        wavenumber grid of the calibrated pixels, where ``S(k)`` is a Gaussian
        source envelope and ``V`` the fringe visibility. Each line advances by
        one :attr:`line_time`, so the mirror moves *within* the frame exactly
        as it does on the real camera -- which is what per-line peak tracking
        is meant to resolve.

        Returns:
            2-D array of irradiance values, shape ``(nlines, num_pixels)``.

        """
        wavenumber = 2 * np.pi / self.wavelength
        center = wavenumber.mean()
        bandwidth = 0.3 * (wavenumber.max() - wavenumber.min())
        envelope = np.exp(-(((wavenumber - center) / bandwidth) ** 2))

        # Line i of this frame was acquired i line periods before its end.
        now = time.perf_counter() - self._start_time
        line_times = now + self.line_time * US_TO_S * np.arange(
            -(self.nlines - 1), 1,
        )
        opds = self.opd_at(line_times)[:, np.newaxis]

        fringes = 1.0 + 0.8 * np.cos(opds * wavenumber)
        noise = self._rng.normal(
            0.0, self.noise_level, (self.nlines, self.num_pixels),
        )

        # Scale to a plausible 12-bit detector range.
        return 2000.0 * envelope * (fringes + noise)
