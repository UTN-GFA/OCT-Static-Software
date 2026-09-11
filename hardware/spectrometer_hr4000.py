# -*- coding: utf-8 -*-
"""
hardware/spectrometer_hr4000.py
Espectrómetro Ocean Optics HR4000 via seabreeze.
"""

import logging
from threading import RLock
from seabreeze.spectrometers import Spectrometer

from .interfaces import SpectrometerInterface

logger = logging.getLogger(__name__)


class HR4000Spectrometer(SpectrometerInterface):
    """
    Control del HR4000 via seabreeze.

    Dark correction
    ---------------
    Usa los electric dark pixels del CCD (Toshiba TCD1304AP):
    píxeles ópticamente enmascarados cuyo promedio se resta
    de toda la lectura. Esto elimina el offset por corriente
    oscura de forma precisa, sin depender del mínimo de la señal.
    La corrección la hace seabreeze internamente al llamar
    ``intensities(correct_dark_counts=True)``.

    Nonlinearity correction
    -----------------------
    El HR4000 almacena coeficientes de corrección de no-linealidad
    en su EEPROM (polinomio de hasta orden 7, calibrado de fábrica).
    Seabreeze los aplica al llamar
    ``intensities(correct_nonlinearity=True)``.
    Corrige la respuesta no-lineal del CCD, importante para que
    la FFT del pipeline OCT trabaje con intensidades proporcionales
    a la señal óptica real.

    """

    def __init__(self):
        self._spec = None
        self._connected = False
        self._io_lock = RLock()
        self._dark_enabled = False
        self._nonlinearity_enabled = False

    def capabilities(self) -> dict:
        return {
            "exposure_control": True,
            "dark_correction": True,
            "nonlinearity_correction": True,
            "wavelengths_nm": True,
            "reconnect": True,
        }

    def connect(self) -> bool:
        if self._spec is None:
            try:
                self._spec = Spectrometer.from_first_available()
            except Exception as e:
                logger.error("HR4000: no se pudo conectar: %s", e)
                return False
        self._connected = self._spec is not None
        return self._connected

    def _invalidate(self):
        """Marcar conexión como perdida tras error de comunicación."""
        if self._connected:
            logger.error("HR4000: comunicación perdida — dispositivo desconectado")
        self._connected = False
        if self._spec is not None:
            try:
                self._spec.close()
            except Exception:
                pass
        self._spec = None

    @property
    def is_connected(self) -> bool:
        return self._connected and self._spec is not None

    def set_exposure_ms(self, value: float) -> None:
        with self._io_lock:
            if self._spec is None:
                raise RuntimeError("HR4000: no conectado")
            try:
                self._spec.integration_time_micros(int(value * 1000))
            except Exception as e:
                self._invalidate()
                raise RuntimeError(f"HR4000: comunicación perdida ({e})")

    def set_dark_correction(self, enabled: bool) -> None:
        self._dark_enabled = bool(enabled)

    def set_nonlinearity_correction(self, enabled: bool) -> None:
        self._nonlinearity_enabled = bool(enabled)

    def read(self):
        with self._io_lock:
            if not self._connected or self._spec is None:
                raise RuntimeError("HR4000: no conectado")

            try:
                wavelengths_nm = self._spec.wavelengths()
                intensities = self._spec.intensities(
                    correct_dark_counts=self._dark_enabled,
                    correct_nonlinearity=self._nonlinearity_enabled,
                )
            except Exception as e:
                self._invalidate()
                raise RuntimeError(f"HR4000: comunicación perdida ({e})")

            if wavelengths_nm is None or intensities is None:
                raise RuntimeError("HR4000: espectrómetro no responde")

            return wavelengths_nm, intensities

    def close(self) -> None:
        self._connected = False
        if self._spec is not None:
            try:
                self._spec.close()
            except Exception:
                pass
        self._spec = None

    def reconnect(self) -> bool:
        """
        Reconexión en caliente tras desconexión USB.

        Cierra el handle stale (seabreeze no detecta la desconexión
        por sí solo) y fuerza un from_first_available() limpio.
        """
        self.close()
        return self.connect()

    def device_info(self) -> dict:
        if self._spec is None:
            return {}
        try:
            return {
                "model": self._spec.model,
                "serial": self._spec.serial_number,
            }
        except Exception:
            self._invalidate()
            return {}
