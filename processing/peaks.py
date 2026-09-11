# -*- coding: utf-8 -*-
"""
processing/peaks.py
Peak Engine: pico dominante por ventana.

Medición principal (PeakEngine)
-------------------------------
El pico dominante de una ventana se define como el máximo del
módulo del perfil axial dentro del rango de la ventana:

    idx = argmax(|profile|)
    depth = depth_axis_m[idx]
    amplitud = |profile|[idx]

Sin detección de picos, sin umbrales ni hiperparámetros. La
interpretación (¿hay señal o es ruido?) queda del lado del
análisis, usando la amplitud guardada.

Inspección de reflexiones dominantes (analyze_profile_peaks)
------------------------------------------------------------
Herramienta de análisis bajo demanda, fuera del pipeline:
identifica las N reflexiones más importantes de un perfil
(máximos locales con separación mínima, ordenados por amplitud).
La usa la GUI para interpretar físicamente perfiles globales
(superficie, interfaces, reflexiones múltiples, rebotes). No
forma parte de ProcessingResult ni del formato de guardado.
"""

import logging
import numpy as np
from scipy.signal import find_peaks as _find_peaks

from .models import WindowResult, PeakData

logger = logging.getLogger(__name__)


# ==================================================================
# Medición principal — pico dominante
# ==================================================================

class PeakEngine:
    """
    Calcula el pico dominante (máximo del módulo) de cada ventana.
    """

    def process(self, window_id: int, result: WindowResult) -> PeakData:
        """
        Pico dominante de un WindowResult.

        Parámetros
        ----------
        window_id : int   (0-based)
        result    : WindowResult

        Retorna
        -------
        PeakData con depth_m/amplitude del máximo, o None/None
        si el perfil está vacío o no contiene valores finitos.
        """
        mag = np.abs(result.profile)

        if mag.size == 0:
            return PeakData(window_id=window_id)

        # Ignorar NaN/Inf sin copiar: si todo es no-finito, no hay pico.
        if not np.all(np.isfinite(mag)):
            if not np.any(np.isfinite(mag)):
                return PeakData(window_id=window_id)
            idx = int(np.nanargmax(np.where(np.isfinite(mag), mag, np.nan)))
        else:
            idx = int(np.argmax(mag))

        return PeakData(
            window_id=window_id,
            depth_m=float(result.depth_axis_m[idx]),
            amplitude=float(mag[idx]),
        )

    def process_all(self, window_results: dict) -> dict:
        """
        Pico dominante de todos los WindowResult.

        Retorna Dict[int, PeakData]
        """
        return {wid: self.process(wid, wr) for wid, wr in window_results.items()}


# ==================================================================
# Inspección de reflexiones dominantes — herramienta bajo demanda
# ==================================================================

def analyze_profile_peaks(
    depth_axis_m: np.ndarray,
    magnitude: np.ndarray,
    n_peaks: int = 6,
    min_separation_m: float = 15e-6,
):
    """
    Identificar las N reflexiones dominantes de un perfil.

    Criterio: máximos locales separados al menos min_separation_m,
    ordenados por amplitud descendente, truncados a los n_peaks más
    altos. Pensada para interpretación física de perfiles globales
    (superficie, interfaces, reflexiones múltiples, rebotes), no
    para detección exhaustiva de picos.

    La separación mínima evita que los lóbulos laterales de un pico
    fuerte ocupen los N lugares antes que reflexiones débiles reales:
    dos máximos más cercanos que varias resoluciones axiales son
    estructura del mismo pico, no reflexiones distintas. El default
    (15 µm) es varias veces la resolución axial típica del sistema.

    Parámetros
    ----------
    depth_axis_m : np.ndarray (N,) metros
    magnitude : np.ndarray (N,) módulo del perfil
    n_peaks : int
        Cantidad máxima de reflexiones a devolver.
    min_separation_m : float
        Separación mínima entre reflexiones (metros).

    Retorna
    -------
    (depth_m, amplitudes) : Tuple[np.ndarray, np.ndarray]
        Ordenados por amplitud descendente. Vacíos si no hay picos.
    """
    empty = (np.array([]), np.array([]))

    mag = np.asarray(magnitude, dtype=float)
    depth_axis_m = np.asarray(depth_axis_m, dtype=float)
    if mag.size == 0 or depth_axis_m.size != mag.size:
        return empty

    bad = ~np.isfinite(mag)
    if np.any(bad):
        mag = mag.copy()
        mag[bad] = 0.0

    if np.max(mag) <= 0:
        return empty

    # min_separation en samples
    if depth_axis_m.size > 1:
        depth_spacing_m = abs(float(np.mean(np.diff(depth_axis_m))))
        dist_samples = (
            max(1, int(min_separation_m / depth_spacing_m))
            if depth_spacing_m > 0
            else 1
        )
    else:
        dist_samples = 1

    try:
        peaks, _ = _find_peaks(mag, distance=dist_samples)
    except Exception:
        logger.exception("analyze_profile_peaks: find_peaks falló")
        return empty

    if peaks.size == 0:
        return empty

    order = np.argsort(mag[peaks])[::-1][: max(1, int(n_peaks))]
    peaks = peaks[order]
    return depth_axis_m[peaks], mag[peaks]
