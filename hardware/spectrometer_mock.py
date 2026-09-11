# -*- coding: utf-8 -*-
"""
hardware/spectrometer_mock.py
Espectrómetro simulado para desarrollo y testing.
"""

import logging
import time
import numpy as np

from .interfaces import SpectrometerInterface
from constants import DEFAULT_EXPOSURE_MS, DEFAULT_N_PIXELS

logger = logging.getLogger(__name__)


class Reflector:
    """Un reflector sintético en el dominio axial."""
    def __init__(self, depth_m: float, amplitude: float = 1.0):
        self.depth_m = depth_m
        self.amplitude = amplitude


class MockSpectrometer(SpectrometerInterface):
    """
    Espectrómetro simulado.

    Parámetros
    ----------
    wavelength_min_nm, wavelength_max_nm : float
        Rango espectral en nm.
    n_pixels : int
        Número de píxeles del detector.
    reflectors : list[Reflector]
        Reflectores sintéticos. Default: 2 reflectores.
    noise_level : float
        Nivel de ruido relativo (0 = sin ruido).
    dark_counts : float o ndarray
        Offset oscuro simulado, escalar o un valor por píxel.
    """

    def __init__(self,
                 wavelength_min_nm: float = 780.0,
                 wavelength_max_nm: float = 920.0,
                 n_pixels: int = DEFAULT_N_PIXELS,
                 reflectors: list = None,
                 noise_level: float = 0.02,
                 exposure_ms: float = DEFAULT_EXPOSURE_MS,
                 dark_counts: float = 0.0):

        self.wavelength_min_nm = wavelength_min_nm
        self.wavelength_max_nm = wavelength_max_nm
        self.n_pixels = n_pixels
        self.noise_level = noise_level
        self._exposure_ms = exposure_ms
        self._connected = False
        self._dark_enabled = False
        dark = np.asarray(dark_counts, dtype=float)
        if dark.ndim > 1 or (dark.ndim == 1 and dark.size != n_pixels):
            raise ValueError(
                "dark_counts debe ser escalar o tener un valor por píxel"
            )
        self._dark_counts = dark.copy()

        if reflectors is None:
            self.reflectors = [
                Reflector(depth_m=0.5e-3, amplitude=1.0),
                Reflector(depth_m=1.2e-3, amplitude=0.6),
            ]
        else:
            self.reflectors = reflectors

        self._wavelengths_nm = np.linspace(
            wavelength_min_nm, wavelength_max_nm, n_pixels
        )

    # ----------------------------------------------------------
    # SpectrometerInterface
    # ----------------------------------------------------------
    def capabilities(self) -> dict:
        return {
            "exposure_control": True,
            "dark_correction": True,
            "nonlinearity_correction": False,
            "wavelengths_nm": True,
            "reconnect": True,
        }

    def connect(self) -> bool:
        self._connected = True
        logger.info("MockSpectrometer conectado (simulación)")
        return True

    def set_exposure_ms(self, value: float) -> None:
        self._exposure_ms = value

    def set_dark_correction(self, enabled: bool) -> None:
        self._dark_enabled = bool(enabled)

    def set_nonlinearity_correction(self, enabled: bool) -> None:
        if enabled:
            logger.debug(
                "MockSpectrometer: nonlinearity correction activada "
                "(no-op en simulación, señal ya es lineal)"
            )

    def device_info(self) -> dict:
        return {
            "model": "MockSpectrometer",
            "serial": "MOCK-0000",
        }

    def read(self):
        if not self._connected:
            raise RuntimeError(
                "MockSpectrometer: read() llamado sin conectar")

        time.sleep(min(self._exposure_ms / 1000.0, 0.05))

        intensities = self._generate_interferogram()

        if self._dark_enabled:
            intensities = np.maximum(intensities - self._dark_counts, 0.0)

        # Nonlinearity: no-op en simulación (la señal sintética
        # ya es lineal, no hay respuesta de CCD que corregir).

        return self._wavelengths_nm.copy(), intensities

    def close(self) -> None:
        self._connected = False
        logger.info("MockSpectrometer desconectado")

    # ----------------------------------------------------------
    # Generación del interferograma
    # ----------------------------------------------------------
    def _generate_interferogram(self) -> np.ndarray:
        wavelengths_m = self._wavelengths_nm * 1e-9
        k = 2 * np.pi / wavelengths_m

        k_center = k.mean()
        k_sigma = (k.max() - k.min()) / 4
        envelope = np.exp(-0.5 * ((k - k_center) / k_sigma) ** 2)

        interference = np.zeros_like(k)
        for r in self.reflectors:
            interference += r.amplitude * np.cos(2 * k * r.depth_m)

        signal = envelope * (1.0 + 0.3 * interference)

        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level * signal.max(),
                                     size=len(signal))
            signal += noise

        signal = np.clip(signal, 0, None)
        signal = signal / signal.max() * 50000

        return signal
