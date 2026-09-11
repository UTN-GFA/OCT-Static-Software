# -*- coding: utf-8 -*-
"""
acquisition/acquisition.py
Capa de adquisición OCT.

Contiene:
  - ScanConfig: configuración del barrido espacial
  - RawPointData: datos crudos de un punto
  - AcquisitionReader: lógica de lectura de espectros
"""

import logging
import time
from dataclasses import dataclass
from typing import List

import numpy as np

from constants import (
    DEFAULT_AXIS_ORDER,
    DEFAULT_EXPOSURE_MS,
    DEFAULT_SCAN_MODE,
    DEFAULT_SETTLING_TIME_S,
    frange,
    count_range,
)
from hardware.interfaces import SpectrometerInterface

logger = logging.getLogger(__name__)


# ==================================================================
# ScanConfig — fuente única de verdad para barridos
# ==================================================================


@dataclass
class ScanConfig:
    """
    Configuración completa de un barrido espacial.

    axis_order define el orden de anidamiento del barrido.
    Se lee de izquierda a derecha como rápido → lento:
      "XYZ" → X rápido, Y medio, Z lento
      "YXZ" → Y rápido, X medio, Z lento
      "ZXY" → Z rápido, X medio, Y lento

    Snake se aplica al eje rápido (el primero): filas impares
    se recorren en dirección inversa.

    Solo se iteran los ejes activos (use_x/use_y/use_z).
    """

    # ── Ejes activos ─────────────────────────────────────────
    use_x: bool = False
    use_y: bool = False
    use_z: bool = False

    # ── Rangos por eje (mm) ──────────────────────────────────
    x_start_mm: float = 0.0
    x_end_mm: float = 0.0
    x_step_mm: float = 0.1

    y_start_mm: float = 0.0
    y_end_mm: float = 0.0
    y_step_mm: float = 0.1

    z_start_mm: float = 0.0
    z_end_mm: float = 0.0
    z_step_mm: float = 0.1

    # ── Tiempos ──────────────────────────────────────────────
    settling_time_s: float = DEFAULT_SETTLING_TIME_S  # segundos, post-movimiento
    exposure_ms: float = DEFAULT_EXPOSURE_MS  # tiempo de integración

    # ── Modo de barrido ──────────────────────────────────────
    scan_mode: str = DEFAULT_SCAN_MODE  # "snake" | "raster"

    # ── Orden de ejes (rápido → lento) ───────────────────────
    axis_order: str = DEFAULT_AXIS_ORDER

    # ── Helpers internos ─────────────────────────────────────

    def _axis_params(self, axis_name: str):
        """Retorna (use, start, end, step) para un eje por nombre."""
        a = axis_name.upper()
        if a == "X":
            return self.use_x, self.x_start_mm, self.x_end_mm, self.x_step_mm
        elif a == "Y":
            return self.use_y, self.y_start_mm, self.y_end_mm, self.y_step_mm
        elif a == "Z":
            return self.use_z, self.z_start_mm, self.z_end_mm, self.z_step_mm
        raise ValueError(f"Eje desconocido: {axis_name}")

    # ── Propiedades derivadas ────────────────────────────────

    @property
    def active_axes(self) -> List[str]:
        """Ejes activos en el orden de axis_order (rápido → lento)."""
        return [a for a in self.axis_order.upper() if self._axis_params(a)[0]]

    @property
    def planned_points(self) -> int:
        """Cantidad de puntos planificados para el barrido."""
        total = 1
        for a in self.active_axes:
            _, start, end, step = self._axis_params(a)
            total *= count_range(start, end, step)
        return total

    def iter_points(self, inactive_pos=None):
        """
        Generador de tuplas (x_mm, y_mm, z_mechanical_mm) en el orden
        del barrido.

        axis_order se expresa rápido → lento. El anidamiento interno
        invierte ese orden para generar:
          eje más lento = outer loop
          eje más rápido = inner loop (con snake)

        Parámetros
        ----------
        inactive_pos : dict, optional
            Posiciones fijas para ejes no activos (ej: {"Y": 2.5}).
            Se usan en vez de 0.0 para reflejar la posición real
            del motor durante el scan.

        Yields
        ------
        (x_mm, y_mm, z_mechanical_mm) : float, float, float
        """
        axes = list(reversed(self.active_axes))
        defaults = {"X": 0.0, "Y": 0.0, "Z": 0.0}
        if inactive_pos:
            defaults.update(inactive_pos)

        if len(axes) == 0:
            yield (defaults["X"], defaults["Y"], defaults["Z"])
            return

        # Construir parámetros de rango para cada eje activo
        ranges = []
        for a in axes:
            _, start, end, step = self._axis_params(a)
            ranges.append((a, start, end, step))

        yield from self._iter_nested(ranges, row_counter=[0], defaults=defaults)

    def _iter_nested(self, ranges, row_counter, defaults, nesting_level=0):
        """Generación recursiva. El nivel más profundo es el eje rápido."""
        axis_name, start, end, step = ranges[nesting_level]
        is_fast = nesting_level == len(ranges) - 1

        if is_fast:
            # Construir una única grilla para el eje rápido. Snake sólo
            # invierte el orden de esa grilla; no genera otra progresión
            # desde `end`, que podría quedar desplazada si el paso no
            # divide exactamente el rango.
            values = list(frange(start, end, step))
            if self.scan_mode == "snake" and row_counter[0] % 2 == 1:
                values.reverse()

            for val in values:
                pos = dict(defaults)
                pos[axis_name] = val
                yield (pos["X"], pos["Y"], pos["Z"])

            row_counter[0] += 1
        else:
            for val in frange(start, end, step):
                for point in self._iter_nested(
                    ranges, row_counter, defaults, nesting_level + 1
                ):
                    x_mm, y_mm, z_mechanical_mm = point
                    if axis_name == "X":
                        yield (val, y_mm, z_mechanical_mm)
                    elif axis_name == "Y":
                        yield (x_mm, val, z_mechanical_mm)
                    else:
                        yield (x_mm, y_mm, val)


# ==================================================================
# RawPointData — salida de AcquisitionReader
# ==================================================================


@dataclass
class RawPointData:
    """
    Datos crudos de un punto espacial.
    Salida del AcquisitionReader, entrada del ProcessingEngine.
    """

    x_mm: float
    y_mm: float
    z_mechanical_mm: float

    wavelengths_nm: np.ndarray  # (N_pixels,)
    spectra: np.ndarray  # (N_spectra, N_pixels), siempre 2D

    timestamp: float = 0.0

    @property
    def n_pixels(self) -> int:
        return self.spectra.shape[1]

    @property
    def spectra_count(self) -> int:
        return self.spectra.shape[0]


# ==================================================================
# AcquisitionReader — lectura de espectros
# ==================================================================


class AcquisitionReader:
    """
    Adquiere espectros válidos en una posición espacial dada.

    Flujo por punto:
        1. Motor llegó y settling mecánico completado (ScanWorker)
        2. READ 1: descartar — integración ocurrió durante el movimiento
        3. WAIT t_exp: nueva integración limpia
        4. READ 2..M+1: M espectros válidos
    """

    def __init__(self, spectrometer: SpectrometerInterface, config: ScanConfig):
        self.spec = spectrometer
        self.config = config

    def update_config(self, config: ScanConfig):
        """Actualizar config en caliente (solo fuera de un scan activo)."""
        self.config = config

    def set_spectrometer(self, spec: SpectrometerInterface):
        """Reemplazar espectrómetro (reconexión en caliente)."""
        self.spec = spec

    def acquire_point(
        self,
        x_mm: float,
        y_mm: float,
        z_mechanical_mm: float,
        m: int,
    ) -> RawPointData:
        """
        Adquirir M espectros válidos en la posición mecánica
        (x_mm, y_mm, z_mechanical_mm).
        """
        if m < 1:
            raise ValueError(f"m debe ser >= 1 (recibió {m})")

        t_exp_s = self.config.exposure_ms / 1000.0

        # READ 1: descarte
        wavelengths_nm, _ = self.spec.read()
        if wavelengths_nm is None:
            raise RuntimeError(
                "Espectrómetro no responde en punto "
                f"({x_mm:.3f}, {y_mm:.3f}, {z_mechanical_mm:.3f}) mm"
            )

        # WAIT: integración limpia
        time.sleep(t_exp_s)

        # READ 2..M+1: espectros válidos
        valid_spectra = []
        wavelengths_nm = None

        for i in range(m):
            wavelengths_i_nm, intensities_i = self.spec.read()

            if wavelengths_i_nm is None or intensities_i is None:
                raise RuntimeError(
                    f"Lectura {i+1}/{m} fallida en "
                    f"({x_mm:.3f}, {y_mm:.3f}, {z_mechanical_mm:.3f}) mm"
                )

            wavelengths_i_nm = np.asarray(wavelengths_i_nm, dtype=np.float64)
            intensities_array = np.asarray(intensities_i, dtype=np.float64)

            if wavelengths_i_nm.ndim != 1 or intensities_array.ndim != 1:
                raise RuntimeError(
                    f"Lectura {i+1}/{m}: wavelengths_nm e intensities deben ser vectores 1-D"
                )
            if len(intensities_array) != len(wavelengths_i_nm):
                raise RuntimeError(
                    f"Lectura {i+1}/{m}: N_pixels inconsistente entre "
                    f"wavelengths_nm e intensities "
                    f"({len(wavelengths_i_nm)} vs {len(intensities_array)})"
                )

            if wavelengths_nm is None:
                wavelengths_nm = wavelengths_i_nm
            elif len(wavelengths_i_nm) != len(wavelengths_nm):
                raise RuntimeError(
                    f"Lectura {i+1}/{m}: N_pixels inconsistente "
                    f"({len(wavelengths_i_nm)} vs {len(wavelengths_nm)})"
                )

            valid_spectra.append(intensities_array)

            if i < m - 1:
                time.sleep(t_exp_s)

        return RawPointData(
            x_mm=x_mm,
            y_mm=y_mm,
            z_mechanical_mm=z_mechanical_mm,
            wavelengths_nm=wavelengths_nm,
            spectra=np.stack(valid_spectra),
            timestamp=time.time(),
        )

    def acquire_monitor(self) -> RawPointData:
        """
        Adquisición para Modo Monitor (sin descarte, sin posición).
        Siempre devuelve 1 espectro.
        """
        wavelengths_nm, intensities = self.spec.read()

        if wavelengths_nm is None or intensities is None:
            raise RuntimeError("Espectrómetro no responde en Modo Monitor")

        wavelengths_array_nm = np.asarray(wavelengths_nm, dtype=np.float64)
        intensities_array = np.asarray(intensities, dtype=np.float64)
        if wavelengths_array_nm.ndim != 1 or intensities_array.ndim != 1:
            raise RuntimeError("Monitor: wavelengths_nm e intensities deben ser vectores 1-D")
        if len(wavelengths_array_nm) != len(intensities_array):
            raise RuntimeError(
                "Monitor: longitud de wavelengths_nm e intensities inconsistente "
                f"({len(wavelengths_array_nm)} vs {len(intensities_array)})"
            )

        return RawPointData(
            x_mm=0.0,
            y_mm=0.0,
            z_mechanical_mm=0.0,
            wavelengths_nm=wavelengths_array_nm,
            spectra=intensities_array[np.newaxis, :],
            timestamp=time.time(),
        )
