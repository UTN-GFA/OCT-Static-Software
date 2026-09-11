# -*- coding: utf-8 -*-
"""
processing/engine.py
Processing Engine: orquestador del pipeline OCT.

Pipeline (por medición i en [0, M)):
    RawPointData
        → PreprocessingEngine  → SpectraKSpace (batch, shape (M, N_k))
        → para cada i:
            → TransformEngine   → Dict[int, WindowResult]
              (en modo global, el perfil global se slicea a las
               ventanas activas antes de buscar picos)
            → PeakEngine        → Dict[int, PeakData]
              (pico dominante = máximo del módulo en la ventana)
        → ProcessingResult (con M window_results y M peaks)

Correspondencia 1:1 estricta:
    raw.spectra[i]  ↔  k_space.spectra_k[i]
                    ↔  window_results[i][win_id]
                    ↔  peaks[i][win_id]

"""

import logging
import numpy as np

from .models import ProcessingConfig
from .models import ProcessingResult, SpectraKSpace, WindowResult
from .preprocessing import PreprocessingEngine
from .transform import TransformEngine
from .peaks import PeakEngine

logger = logging.getLogger(__name__)


class ProcessingEngine:
    """
    Punto de entrada único para todo el procesamiento OCT.

    La GUI llama a engine.process(raw_point_data) y recibe
    un ProcessingResult listo para graficar y guardar.
    """

    def __init__(self, config: ProcessingConfig = None):
        if config is None:
            config = ProcessingConfig()
        self.config = config
        self._build_stages()

    def update_config(self, config: ProcessingConfig):
        """Actualizar configuración en caliente (desde GUI)."""
        self.config = config
        self._build_stages()

    def process_raw(
        self, wavelengths_nm: np.ndarray, spectra: np.ndarray
    ) -> ProcessingResult:
        """
        Procesar desde arrays crudos (wavelengths_nm + spectra).

        Punto de entrada alternativo para cuando no se dispone
        de un RawPointData completo (ej: reprocesamiento offline,
        modo monitor simplificado).

        Parámetros
        ----------
        wavelengths_nm : (N_pixels,)
        spectra : (N_spectra, N_pixels) o (N_pixels,)
            Si 1D, se promueve a (1, N_pixels).
        """
        spectra_2d = np.atleast_2d(spectra)
        return self._run_pipeline(wavelengths_nm, spectra_2d)

    def process(self, raw) -> ProcessingResult:
        """
        Procesar un RawPointData completo.

        Parámetros
        ----------
        raw : RawPointData
            Datos crudos de un punto espacial (acquisition.models).
            raw.spectra shape: (N_spectra, N_pixels)

        Retorna
        -------
        ProcessingResult
        """
        return self._run_pipeline(raw.wavelengths_nm, raw.spectra)

    # ----------------------------------------------------------
    # Pipeline interno
    # ----------------------------------------------------------
    def _run_pipeline(
        self, wavelengths_nm: np.ndarray, spectra_2d: np.ndarray
    ) -> ProcessingResult:
        """
        Pipeline completo, sin promedios, procesando cada medición
        de forma independiente:

            spectra[i] → Preprocessing → Transform → Peaks   (para i en [0, M))

        La correspondencia 1:1 entre i, window_results[i] y peaks[i]
        está garantizada por construcción (mismo índice de lista).
        """
        # ── 1. Preprocessing: λ → k uniforme (batch, shape (M, N_k)) ──
        kspace_batch = self._pre.process_batch(wavelengths_nm, spectra_2d)

        # Forzar 2D para que la iteración por índice funcione igual
        # con M=1 o M>1 (consistencia del pipeline, sin ramas especiales).
        spectra_k_all = np.atleast_2d(kspace_batch.spectra_k)
        m = spectra_k_all.shape[0]

        is_windowed = self.config.use_windows

        # ── 2. Loop por medición: transform + peaks por separado ──
        window_results_per_m: list = []
        peaks_per_m: list = []

        for i in range(m):
            kspace_i = SpectraKSpace(
                k_axis=kspace_batch.k_axis,
                spectra_k=spectra_k_all[i],  # 1D: (N_k,)
            )
            wr_i = self._transform.process(kspace_i, self._pre.fs_k)

            if not is_windowed:
                # Modo global: CZT sobre rango completo (key -1).
                # Se slicea el perfil global a cada ventana activa;
                # el pico dominante se busca dentro de cada slice.
                global_wr_i = wr_i.get(-1)
                if global_wr_i is not None:
                    wr_i = self._slice_global_to_windows(global_wr_i)

            # Pico dominante por ventana: argmax(|profile|).
            pk_i = self._peaks.process_all(wr_i)

            window_results_per_m.append(wr_i)
            peaks_per_m.append(pk_i)

        # ── 3. Empaquetar resultado ──────────────────────────────
        # k_space conserva TODOS los espectros (2D) para mantener
        # la correspondencia 1:1 espectro↔perfil↔picos.
        kspace_full = SpectraKSpace(
            k_axis=kspace_batch.k_axis,
            spectra_k=spectra_k_all,  # (M, N_k)
        )

        return ProcessingResult(
            k_space=kspace_full,
            is_windowed=is_windowed,
            window_results=window_results_per_m,
            peaks=peaks_per_m,
        )

    # ----------------------------------------------------------
    # Slicing global → ventanas
    # ----------------------------------------------------------
    def _slice_global_to_windows(self, global_wr) -> dict:
        """
        En modo global, expone el perfil dentro de cada ventana activa.
        """
        active_windows = self.config.active_windows

        if not active_windows:
            return {0: global_wr}

        result = {}
        for win in active_windows:
            mask = (global_wr.depth_axis_m >= win.depth_min_m) & (global_wr.depth_axis_m <= win.depth_max_m)
            if np.any(mask):
                result[win.index] = WindowResult(
                    depth_axis_m=global_wr.depth_axis_m[mask],
                    profile=global_wr.profile[mask],
                )
            else:
                result[win.index] = WindowResult(
                    depth_axis_m=np.array([]),
                    profile=np.array([], dtype=complex),
                )
        return result

    # ----------------------------------------------------------
    # Helpers de conveniencia
    # ----------------------------------------------------------
    @staticmethod
    def theoretical_resolution_um(
        wavelength_min_nm: float, wavelength_max_nm: float
    ) -> float:
        """Resolución axial teórica (FWHM) en µm."""
        wavelength_min_m = wavelength_min_nm * 1e-9
        wavelength_max_m = wavelength_max_nm * 1e-9
        wavelength_center_m = (wavelength_min_m + wavelength_max_m) / 2
        res = (2 * np.log(2) / np.pi) * wavelength_center_m**2 / (
            wavelength_max_m - wavelength_min_m
        )
        return res * 1e6

    @staticmethod
    def theoretical_depth_range_mm(
        wavelength_min_nm: float, wavelength_max_nm: float, n_pixels: int
    ) -> float:
        """Rango axial máximo no ambiguo en mm.

        Para I(k) ~ cos(2·z·k), el límite de Nyquist del eje de
        profundidad es z_max = π / (2·Δk), con Δk dado por el
        espaciamiento del eje k muestreado.
        """
        if wavelength_min_nm <= 0 or wavelength_max_nm <= wavelength_min_nm:
            raise ValueError("Se requiere 0 < wavelength_min_nm < wavelength_max_nm")
        if n_pixels < 2:
            raise ValueError("n_pixels debe ser >= 2")

        wavelength_min_m = wavelength_min_nm * 1e-9
        wavelength_max_m = wavelength_max_nm * 1e-9
        delta_k_total = (
            2 * np.pi / wavelength_min_m - 2 * np.pi / wavelength_max_m
        )
        dk = delta_k_total / (n_pixels - 1)
        return (np.pi / (2 * dk)) * 1e3

    # ----------------------------------------------------------
    # Internals
    # ----------------------------------------------------------
    def _build_stages(self):
        self._pre = PreprocessingEngine(self.config)
        self._transform = TransformEngine(self.config)
        self._peaks = PeakEngine()
