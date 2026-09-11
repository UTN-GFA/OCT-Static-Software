# -*- coding: utf-8 -*-
"""
processing/transform.py
Transform Engine: CZT únicamente.
"""

import logging
import numpy as np
from scipy.signal import zoom_fft

from .models import ProcessingConfig
from .models import SpectraKSpace, WindowResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# CZT — wrapper sobre scipy.signal.zoom_fft
# ------------------------------------------------------------------
def czt(
    signal: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
    fs_k: float,
    n_points: int,
):
    """
    Chirp Z-Transform via scipy.signal.zoom_fft.

    Evalúa la Z-transform de `signal` en n_points uniformemente
    espaciados entre depth_min_m y depth_max_m, con frecuencia de
    muestreo fs_k.

    Parámetros
    ----------
    signal : (N,) array
        Señal de entrada (real o compleja).
    depth_min_m : float
        Profundidad mínima en metros.
    depth_max_m : float
        Profundidad máxima en metros.
    fs_k : float
        Frecuencia de muestreo en k-space (π/dk).
    n_points : int
        Número de puntos de salida (exacto, sin redondeo).

    Retorna
    -------
    z : (n_points,) ndarray complejo
        Transformada CZT.
    depth_axis_m : (n_points,) ndarray float
        Eje de profundidad física correspondiente (metros).
    """
    if n_points <= 0:
        raise ValueError(f"CZT: n_points debe ser > 0 (recibió {n_points})")
    if depth_max_m <= depth_min_m:
        raise ValueError(
            "CZT: depth_max_m debe ser > depth_min_m "
            f"(depth_min_m={depth_min_m}, depth_max_m={depth_max_m})"
        )
    if fs_k <= 0:
        raise ValueError(f"CZT: fs_k debe ser > 0 (recibió {fs_k})")

    z = zoom_fft(
        signal,
        [depth_min_m, depth_max_m],
        m=n_points,
        fs=fs_k,
        endpoint=False,
    )
    depth_axis_m = np.linspace(depth_min_m, depth_max_m, n_points, endpoint=False)

    return z, depth_axis_m


# ------------------------------------------------------------------
# Transform Engine
# ------------------------------------------------------------------
class TransformEngine:
    """
    Aplica CZT a espectros en k-space.

    Modos (mutuamente excluyentes):
    - Global  : CZT en [depth_min_global_m, depth_max_global_m] con global_profile_samples puntos.
    - Ventanas: CZT en cada ventana habilitada de config.windows.
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config

    def process(self, kspace: SpectraKSpace, fs_k: float) -> dict:
        """
        Aplicar CZT al espectro en k-space.

        Parámetros
        ----------
        kspace : SpectraKSpace
            Salida del PreprocessingEngine para UNA medición. spectra_k
            debe ser 1D — el ProcessingEngine es responsable de llamar
            a este método una vez por cada medición i en [0, M).
        fs_k : float
            Frecuencia de muestreo en k-space (PreprocessingEngine.fs_k).

        Retorna
        -------
        Dict[int, WindowResult]
            key -1  → resultado global
            key 0..N → resultado por ventana

        Raises
        ------
        ValueError si spectra_k no es 1D.
        """
        s = kspace.spectra_k
        if s.ndim != 1:
            raise ValueError(
                f"TransformEngine.process espera spectra_k 1D "
                f"(recibió ndim={s.ndim}, shape={s.shape}). "
                f"El ProcessingEngine debe iterar sobre M y pasar "
                f"un espectro por vez — no se permite promediar aquí."
            )

        if self.config.use_windows:
            return self._process_windows(s, fs_k)
        else:
            return self._process_global(s, fs_k)

    # ----------------------------------------------------------
    # Modo global
    # ----------------------------------------------------------
    def _process_global(self, s: np.ndarray, fs_k: float) -> dict:
        depth_min_m = self.config.depth_min_global_m
        depth_max_m = self.config.depth_max_global_m
        n_profile_samples = self.config.global_profile_samples

        profile, depth_axis_m = czt(
            s, depth_min_m, depth_max_m, fs_k, n_profile_samples
        )
        return {-1: WindowResult(depth_axis_m=depth_axis_m, profile=profile)}

    # ----------------------------------------------------------
    # Modo ventanas
    # ----------------------------------------------------------
    def _process_windows(self, s: np.ndarray, fs_k: float) -> dict:
        results = {}

        for win in self.config.active_windows:
            try:
                profile, depth_axis_m = czt(
                    s, win.depth_min_m, win.depth_max_m, fs_k, win.window_profile_samples
                )
                results[win.index] = WindowResult(
                    depth_axis_m=depth_axis_m, profile=profile
                )
            except Exception as e:
                logger.error("TransformEngine: error en ventana %d: %s", win.index, e)

        return results
