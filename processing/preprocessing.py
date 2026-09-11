# -*- coding: utf-8 -*-
"""
processing/preprocessing.py
Preprocessing Engine: λ → k uniforme.


Convención fs_k:
    Señal OCT: I(k) ~ cos(2·z·k)  →  frecuencia discreta ω₀ = 2·z·dk
    Para que la CZT devuelva picos en f = z (metros):
        fs_k = π / dk
"""

import logging
import numpy as np

from .models import ProcessingConfig
from .models import SpectraKSpace

logger = logging.getLogger(__name__)


def depth_to_opd_m(depth_m):
    """Convertir profundidad física en OPD de reflexión.

    En geometría de reflexión, la diferencia de camino óptico es el
    doble de la profundidad física: OPD = 2 · depth.
    """
    return 2.0 * np.asarray(depth_m)


class PreprocessingEngine:
    """
    Convierte espectros de λ (nm) a k uniforme (rad/m).

    El eje k se construye una sola vez y se reutiliza en todas
    las llamadas (cache). Se reconstruye solo si cambian los
    wavelengths o la config.
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._cache_wavelengths_m = None
        self._cache_k_axis = None
        self._cache_k_min = None
        self._cache_k_max = None
        self._cache_n_k = None

    # ----------------------------------------------------------
    # API pública
    # ----------------------------------------------------------
    def process_batch(
        self, wavelengths_nm: np.ndarray, spectra: np.ndarray
    ) -> SpectraKSpace:
        """
        Convierte un batch de espectros a k-space uniforme.

        Parámetros
        ----------
        wavelengths_nm : (N_pixels,)
        spectra        : (N_spectra, N_pixels)

        Retorna
        -------
        SpectraKSpace con spectra_k (N_spectra, N_k)
        """
        k_axis = self._get_k_axis(wavelengths_nm)
        result = np.stack(
            [self._interpolate(wavelengths_nm, s, k_axis) for s in spectra]
        )
        return SpectraKSpace(k_axis=k_axis, spectra_k=result)

    # ----------------------------------------------------------
    # Internals
    # ----------------------------------------------------------
    def _get_k_axis(self, wavelengths_nm: np.ndarray) -> np.ndarray:
        """Construir (o recuperar del cache) el eje k uniforme."""
        wavelengths_m = wavelengths_nm * 1e-9

        # Reconstruir cache si cambian wavelengths O parámetros de config
        config_changed = (
            self._cache_k_min != self.config.k_min
            or self._cache_k_max != self.config.k_max
            or self._cache_n_k != self.config.n_k
        )
        wavelengths_changed = (
            self._cache_wavelengths_m is None
            or len(self._cache_wavelengths_m) != len(wavelengths_m)
            or not np.allclose(self._cache_wavelengths_m, wavelengths_m, rtol=1e-8)
        )

        if config_changed or wavelengths_changed:
            self._cache_wavelengths_m = wavelengths_m.copy()
            self._cache_k_min = self.config.k_min
            self._cache_k_max = self.config.k_max
            self._cache_n_k = self.config.n_k

            k_raw = 2 * np.pi / wavelengths_m
            k_min = self.config.k_min if self.config.k_min is not None else k_raw.min()
            k_max = self.config.k_max if self.config.k_max is not None else k_raw.max()

            self._cache_k_axis = np.linspace(k_min, k_max, self.config.n_k)
            logger.debug(
                "Eje k reconstruido: k=[%.2f, %.2f] rad/m, N_k=%d",
                k_min,
                k_max,
                self.config.n_k,
            )

        return self._cache_k_axis

    def _interpolate(
        self, wavelengths_nm: np.ndarray, spectrum: np.ndarray, k_axis: np.ndarray
    ) -> np.ndarray:
        """
        Interpolar espectro sobre eje k uniforme.
        SOLO interpolación lineal (doc sección 5).
        NaN/Inf en el espectro de entrada se reemplazan por 0
        antes de interpolar para no propagar basura al pipeline.
        """
        wavelengths_m = wavelengths_nm * 1e-9
        k_raw = 2 * np.pi / wavelengths_m

        s = np.asarray(spectrum, dtype=float)

        # Sanitizar NaN/Inf → reemplazar por 0
        bad = ~np.isfinite(s)
        if np.any(bad):
            n_bad = int(bad.sum())
            logger.warning(
                "Preprocessing: %d valores NaN/Inf reemplazados por 0", n_bad
            )
            s = s.copy()
            s[bad] = 0.0

        # Ordenar por k creciente (wavelengths_nm decrecientes → k creciente)
        sort_idx = np.argsort(k_raw)
        k_sorted = k_raw[sort_idx]
        s_sorted = s[sort_idx]

        # FIX: validar que k es estrictamente creciente.
        # np.interp requiere xp monótono creciente; duplicados o
        # no-monotonía dan resultados indefinidos.
        if not np.all(np.diff(k_sorted) > 0):
            raise ValueError(
                "Preprocessing: eje k no es estrictamente creciente "
                "tras ordenar (wavelengths duplicadas o no monótonas)."
            )

        # Interpolación lineal — única permitida
        result = np.interp(k_axis, k_sorted, s_sorted)

        # Verificación post-interpolación
        if not np.all(np.isfinite(result)):
            logger.warning("Preprocessing: NaN/Inf post-interpolación, forzando a 0")
            result = np.where(np.isfinite(result), result, 0.0)

        return result

    # ----------------------------------------------------------
    # Propiedades derivadas del eje k
    # ----------------------------------------------------------
    @property
    def dk(self) -> float:
        """Spacing del eje k (rad/m). Requiere haber procesado al menos un espectro."""
        if self._cache_k_axis is None:
            raise RuntimeError(
                "dk no disponible: procesar al menos un espectro primero"
            )
        return float(np.mean(np.diff(self._cache_k_axis)))

    @property
    def fs_k(self) -> float:
        """
        Parámetro fs para la CZT tal que depth_min_m/depth_max_m
        en metros correspondan
        a profundidades físicas.

        Derivación:
            Señal OCT: I(k) = cos(2·depth₀·k)
            Frecuencia discreta: ω₀ = 2·depth₀·dk
            CZT: pico en depth_axis_m donde 2π·depth_axis_m/fs = ω₀
            Para depth_axis_m = depth₀: fs = π / dk
        """
        return np.pi / self.dk
