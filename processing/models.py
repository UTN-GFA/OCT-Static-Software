# -*- coding: utf-8 -*-
"""
processing/models.py
Estructuras de datos del pipeline de procesamiento OCT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from constants import DEFAULT_K_SAMPLES, DEFAULT_PROFILE_SAMPLES, DEFAULT_Z_MAX_GLOBAL_M


# ==================================================================
# Configuración
# ==================================================================

@dataclass
class WindowConfig:
    """Configuración de una ventana de profundidad."""

    index: int  # 0-based
    depth_min_m: float
    depth_max_m: float
    window_profile_samples: int = 2048
    enabled: bool = True


@dataclass
class ProcessingConfig:
    """
    Configuración completa del pipeline OCT.

    Parámetros
    ----------
    k_min, k_max : Optional[float]
        Rango del eje k (rad/m). Si None, se calculan automáticamente
        desde los wavelengths del primer espectro.
    n_k : int
        Número de puntos del eje k uniforme.
    global_profile_samples : int
        Puntos de la CZT global (modo sin ventanas).
    depth_min_global_m, depth_max_global_m : float
        Rango global de profundidad física (metros).
    measurements_per_point : int
        Cantidad de mediciones válidas por punto espacial (M ≥ 1).
        Cada medición es procesada independientemente (sin promedios)
        y produce un resultado propio (spectra, profile, peaks).
    use_windows : bool
        True → CZT por ventanas. False → CZT global.
    windows : List[WindowConfig]
        Ventanas definidas (se usan solo las habilitadas).
    """

    # ── K-space ───────────────────────────────────────────────────
    k_min: Optional[float] = None
    k_max: Optional[float] = None
    n_k: int = DEFAULT_K_SAMPLES

    # ── CZT global ───────────────────────────────────────────────
    global_profile_samples: int = DEFAULT_PROFILE_SAMPLES
    depth_min_global_m: float = 0.0
    depth_max_global_m: float = DEFAULT_Z_MAX_GLOBAL_M

    # ── Mediciones por punto ─────────────────────────────────────
    measurements_per_point: int = 1

    # ── Modo de CZT ──────────────────────────────────────────────
    use_windows: bool = False

    # ── Ventanas (lista mutable, default vacía) ──────────────────
    windows: List[WindowConfig] = field(default_factory=list)

    # ── Propiedades derivadas ────────────────────────────────────
    @property
    def active_windows(self) -> List[WindowConfig]:
        """Ventanas habilitadas con rango válido."""
        return [w for w in self.windows if w.enabled and w.depth_max_m > w.depth_min_m]


    # ── Factory: la GUI SOLO llama a esto ────────────────────────
    @classmethod
    def from_gui_params(
        cls,
        *,
        windows: List[WindowConfig],
        depth_min_global_m: float = 0.0,
        depth_max_global_m: float = DEFAULT_Z_MAX_GLOBAL_M,
        n_k: int = DEFAULT_K_SAMPLES,
        global_profile_samples: int = DEFAULT_PROFILE_SAMPLES,
        measurements_per_point: int = 1,
        k_min: Optional[float] = None,
        k_max: Optional[float] = None,
    ) -> ProcessingConfig:
        """
        Construir config desde parámetros crudos de la GUI.

        Toda la lógica de derivación (use_windows, rangos de profundidad
        efectivos) vive aquí, no en la GUI.
        """
        active = [w for w in windows if w.enabled and w.depth_max_m > w.depth_min_m]
        use_windows = len(active) > 0

        if use_windows:
            depth_min_m = min(w.depth_min_m for w in active)
            depth_max_m = max(w.depth_max_m for w in active)
        else:
            depth_min_m = depth_min_global_m
            depth_max_m = depth_max_global_m

        return cls(
            k_min=k_min,
            k_max=k_max,
            n_k=n_k,
            global_profile_samples=global_profile_samples,
            depth_min_global_m=depth_min_m,
            depth_max_global_m=depth_max_m,
            measurements_per_point=measurements_per_point,
            use_windows=use_windows,
            windows=windows,
        )


# ==================================================================
# Resultados de cada etapa
# ==================================================================

@dataclass
class SpectraKSpace:
    """
    Espectro(s) interpolados en k uniforme.
    Salida del PreprocessingEngine.
    """

    k_axis: np.ndarray  # (N_k,) rad/m, uniforme
    spectra_k: np.ndarray  # (N_spectra, N_k) o (N_k,) para un solo espectro


@dataclass
class WindowResult:
    """Resultado CZT de una sola ventana (o del rango global)."""

    depth_axis_m: np.ndarray  # (N_z,) metros
    profile: np.ndarray  # (N_z,) complejo


@dataclass
class PeakData:
    """
    Pico dominante de una ventana.

    Se define como el máximo del módulo del perfil axial dentro
    del rango de la ventana. depth_m/amplitude son None si la
    ventana no contiene muestras (perfil vacío o todo no-finito).
    """

    window_id: int  # índice de la ventana (0-based)
    depth_m: Optional[float] = None  # metros (profundidad física del máximo)
    amplitude: Optional[float] = None  # |perfil| en el máximo


# ==================================================================
# Resultado final del pipeline
# ==================================================================

@dataclass
class ProcessingResult:
    """
    Resultado completo del pipeline para un punto.
    Lo que la GUI consume para graficar y guardar.

    Contiene M mediciones procesadas independientemente, indexadas
    por posición en las listas `window_results` y `peaks`:

        window_results[i]  ↔  peaks[i]  ↔  k_space.spectra_k[i]

    Esta correspondencia 1:1 está garantizada por construcción — todas
    las listas tienen largo M y se pueblan en el mismo loop del engine.

    El engine asigna picos a ventanas internamente: la GUI nunca decide
    si el modo es global o por ventanas. Para visualización la GUI puede
    usar result.window_results[0] / result.peaks[0], pero el sistema
    procesa y conserva las M mediciones.
    """

    k_space: SpectraKSpace

    is_windowed: bool = False

    window_results: List[Dict[int, WindowResult]] = field(default_factory=list)

    peaks: List[Dict[int, PeakData]] = field(default_factory=list)

    @property
    def measurements_per_point(self) -> int:
        """Cantidad de mediciones procesadas (M)."""
        return len(self.window_results)
