# -*- coding: utf-8 -*-
"""
storage/saver.py
Guardado de barridos OCT.

Formatos disponibles:
  NPZ (.npz)  — default, sin dependencias externas, acumula en RAM al final
  HDF5 (.h5)  — escritura incremental punto a punto (requiere: pip install h5py)
"""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from constants import (
    DEFAULT_SAVE_DIRECTORY,
    DEFAULT_SAVE_FORMAT,
    DEFAULT_SAVE_PEAKS,
    DEFAULT_SAVE_PROFILE,
    DEFAULT_SAVE_SPECTRA,
    DEFAULT_PROFILE_MODE,
    DEFAULT_POSITION_TOLERANCE_MM,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)

# Keys de parámetros ópticos (solo inputs del sistema).
# Las salidas derivadas —incluida lateral_resolution_um— no forman parte
# del schema persistido; Spot geom. se calcula nuevamente desde estos inputs.
_OPTICS_KEYS = (
    "fiber_diameter_um",
    "wavelength_nm",
    "collimator_focal_length_mm",
    "objective_focal_length_mm",
)

_PERSISTED_OPTICS_KEYS = {
    "fiber_diameter_um": "o_d_fib_um",
    "wavelength_nm": "o_wl_nm",
    "collimator_focal_length_mm": "o_f_col_mm",
    "objective_focal_length_mm": "o_f_obj_mm",
}


# ------------------------------------------------------------------
# SaveConfig — qué datos guardar en el archivo OCT
# ------------------------------------------------------------------
@dataclass
class SaveConfig:
    """
    Qué datos guardar en el archivo.

    profile_mode:
        "modulus" → solo |z| (float64, mitad de espacio)
        "complex" → Re + Im (float64 x2, para reprocesamiento de fase)
    """

    save_spectra: bool = DEFAULT_SAVE_SPECTRA

    save_peaks: bool = DEFAULT_SAVE_PEAKS

    save_profile: bool = DEFAULT_SAVE_PROFILE
    profile_mode: str = DEFAULT_PROFILE_MODE  # "modulus" | "complex"

    @property
    def save_complex(self) -> bool:
        return self.save_profile and self.profile_mode == "complex"


# ------------------------------------------------------------------
# Helpers de filename
# ------------------------------------------------------------------
def sanitize_filename(name: str) -> str:
    """Sanitizar nombre para uso en filesystem."""
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^\w\-.]", "", name)
    return name or "scan"


def generate_filename(
    save_directory: str = DEFAULT_SAVE_DIRECTORY,
    file_format: str = DEFAULT_SAVE_FORMAT,
    sample_name: str = "",
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    prefix = sanitize_filename(sample_name) if sample_name else "scan"
    file_format = file_format.lstrip(".")
    filename = f"{prefix}_{ts}.{file_format}"
    os.makedirs(save_directory, exist_ok=True)
    return os.path.join(save_directory, filename)


# ------------------------------------------------------------------
# OCTDataSaver
# ------------------------------------------------------------------
class OCTDataSaver:
    """
    Escritor de barridos OCT.

    Soporta dos formatos:
      "npz"  — guarda todo al final (sin dependencias externas)
      "h5"   — escritura incremental punto a punto (requiere h5py)
    """

    def __init__(
        self,
        schema_version: str = SCHEMA_VERSION,
    ) -> None:
        self.schema_version = schema_version

        # Estado interno
        self._file_format = "npz"
        self._filepath = None
        self._is_open = False
        self._save_cfg = None
        self._metadata = None
        self._planned_points = 0
        self._next_point_idx = 0
        self._measurements_per_point = 1
        self._n_windows = 1
        self._wavelengths_nm: Optional[np.ndarray] = None

        # HDF5
        self._h5file = None
        self._h5_profiles_created = False
        self._h5_written_points = 0

        # Ciclo de vida temporal del scan:
        #   _start_time se captura en open_scan()
        #   _end_time se captura en close_scan()
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

        # NPZ — buffers en RAM
        self._buf_x_mm: list = []
        self._buf_y_mm: list = []
        self._buf_z_mechanical_mm: list = []
        self._buf_depth_m: list = []
        self._buf_amplitude: list = []
        self._buf_spectra: list = []
        self._buf_profiles: list = []
        self._profile_depth_axes_m: Optional[Dict[int, np.ndarray]] = None

    # ── API principal ─────────────────────────────────────────────────────────

    def open_scan(
        self,
        filepath: str,
        planned_points: int,
        n_pixels: int,
        metadata: Dict[str, Any],
        save_cfg,
        file_format: str = "npz",
        measurements_per_point: int = 1,
        wavelengths_nm: Optional[np.ndarray] = None,
    ) -> None:
        """
        Abrir un scan para escritura.

        Parámetros
        ----------
        measurements_per_point : int
            Cantidad de mediciones por punto (M). Se usa para dimensionar
            los datasets HDF5. NPZ acumula y stackea al final.
        wavelengths_nm : np.ndarray
            Calibración píxel→λ del espectrómetro (nm). Obligatorio en
            Schema 6.0 — se guarda como dataset único por archivo.
        """
        if wavelengths_nm is None:
            raise ValueError(
                "Schema 6.0 requiere el vector wavelengths_nm. "
                "No se puede abrir un scan sin calibración espectral."
            )

        self._file_format = file_format.lower()
        self._save_cfg = save_cfg
        self._metadata = metadata
        self._planned_points = planned_points
        self._next_point_idx = 0
        self._measurements_per_point = max(1, int(measurements_per_point))
        self._wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float64)

        win_cfg = metadata.get("windows", [])
        self._n_windows = max(1, len(win_cfg))

        ext = "h5" if self._file_format == "h5" else "npz"
        if not filepath.endswith(f".{ext}"):
            filepath = filepath + f".{ext}"
        self._filepath = filepath

        # Inicio del ciclo temporal
        self._start_time = metadata.get("start_time", datetime.now())
        self._end_time = None

        # Limpiar buffers NPZ
        self._buf_x_mm.clear()
        self._buf_y_mm.clear()
        self._buf_z_mechanical_mm.clear()
        self._buf_depth_m.clear()
        self._buf_amplitude.clear()
        self._buf_spectra.clear()
        self._buf_profiles.clear()
        self._profile_depth_axes_m = None

        if self._file_format == "h5":
            self._open_h5(filepath, planned_points, n_pixels, metadata, save_cfg)
        else:
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            logger.info("NPZ preparado: %s (%d puntos)", filepath, planned_points)

        self._is_open = True

    def write_point(
        self,
        idx: int,
        x_mm: float,
        y_mm: float,
        z_mechanical_mm: float,
        spectra: Optional[np.ndarray],
        depth_m: Optional[np.ndarray],
        amplitude: Optional[np.ndarray],
        profiles: Optional[List[Dict[int, np.ndarray]]] = None,
        profile_depth_axes_m: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        """
        Escribir un punto. O(1) en HDF5; acumula en RAM en NPZ.

        Parámetros
        ----------
        idx : int
            Índice del punto en el scan (0-based).
        x_mm, y_mm, z_mechanical_mm : float
            Posición mecánica canónica (mm).
        spectra : Optional[np.ndarray]
            Espectros crudos shape (M, N_pixels). None si no se guarda.
        depth_m : Optional[np.ndarray]
            Profundidad del pico dominante, shape (M, N_active). None si no
            se guardan picos.
        amplitude : Optional[np.ndarray]
            Amplitud del pico dominante, shape (M, N_active). None si
            no se guardan picos.
        profiles : Optional[List[Dict[int, np.ndarray]]]
            Lista de largo M. profiles[i] mapea win_id → perfil axial
            (complex o módulo) de la medición i. None si no se guarda.
        """
        if not self.is_open():
            raise RuntimeError(
                "write_point() llamado sin un scan abierto. "
                "Llamar open_scan() primero."
            )
        if idx != self._next_point_idx:
            raise ValueError(
                "write_point() requiere índices secuenciales: "
                f"se esperaba {self._next_point_idx}, se recibió {idx}."
            )
        if idx < 0 or idx >= self._planned_points:
            raise IndexError(
                f"Índice de punto fuera del scan: {idx} "
                f"(planned_points={self._planned_points})."
            )
        if self._file_format == "h5":
            self._write_point_h5(
                idx,
                x_mm,
                y_mm,
                z_mechanical_mm,
                spectra,
                depth_m,
                amplitude,
                profiles,
                profile_depth_axes_m,
            )
        else:
            self._write_point_npz(
                x_mm,
                y_mm,
                z_mechanical_mm,
                spectra,
                depth_m,
                amplitude,
                profiles,
                profile_depth_axes_m,
            )
        self._next_point_idx += 1

    def close_scan(self, aborted: bool = False) -> Optional[str]:
        # Fin del ciclo temporal — antes de delegar al formato.
        self._end_time = datetime.now()

        if self._file_format == "h5":
            result = self._close_h5(aborted)
        else:
            result = self._close_npz(aborted)
        self._is_open = False

        # Renombrar archivo si el scan fue abortado
        if aborted and result and os.path.isfile(result):
            result = self._rename_aborted(result)

        return result

    def is_open(self) -> bool:
        """
        Retorna True si hay un barrido abierto para escritura.
        """
        if self._file_format == "h5":
            return self._h5file is not None
        else:
            return self._is_open and self._filepath is not None

    # ── NPZ — escritura al final ──────────────────────────────────────────────

    def _write_point_npz(
        self,
        x_mm,
        y_mm,
        z_mechanical_mm,
        spectra,
        depth_m,
        amplitude,
        profiles,
        profile_depth_axes_m,
    ):
        if profile_depth_axes_m and self._profile_depth_axes_m is None:
            self._profile_depth_axes_m = profile_depth_axes_m
        self._buf_x_mm.append(x_mm)
        self._buf_y_mm.append(y_mm)
        self._buf_z_mechanical_mm.append(z_mechanical_mm)
        if self._save_cfg.save_peaks and depth_m is not None:
            self._buf_depth_m.append(np.asarray(depth_m, dtype=np.float64))
            self._buf_amplitude.append(np.asarray(amplitude, dtype=np.float64))
        if self._save_cfg.save_spectra and spectra is not None:
            self._buf_spectra.append(spectra)
        if self._save_cfg.save_profile and profiles is not None:
            self._buf_profiles.append(profiles)

    def _close_npz(self, aborted: bool) -> Optional[str]:
        if self._filepath is None:
            return None
        n = len(self._buf_x_mm)
        if n == 0:
            logger.warning("NPZ: sin puntos, no se escribe archivo")
            self._filepath = None
            return None

        save_dict: Dict[str, Any] = {
            "wavelengths_nm": self._wavelengths_nm,
            "x_mm": np.asarray(self._buf_x_mm, dtype=np.float64),
            "y_mm": np.asarray(self._buf_y_mm, dtype=np.float64),
            "z_mm": np.asarray(self._buf_z_mechanical_mm, dtype=np.float64),
        }

        # Picos dominantes — shape (planned_points, M, N_win)
        if self._buf_depth_m:
            save_dict["depth_m"] = np.stack(self._buf_depth_m)
            save_dict["amplitude"] = np.stack(self._buf_amplitude)

        # Espectros — llegan como (M, N_pixels), se stackean a
        # (planned_points, M, N_pixels).
        if self._buf_spectra:
            save_dict["spectra"] = np.stack(self._buf_spectra).astype(np.float64)

        # Perfiles axiales — numeración secuencial (w0, w1, ...).
        # Keys del dict ya son secuenciales (renumeradas por ScanController).
        if self._buf_profiles:
            is_complex = self._save_cfg.save_complex
            measurements_per_point = self._measurements_per_point

            for win_i in range(self._n_windows):
                max_len = 0
                for pt_list in self._buf_profiles:
                    for meas_dict in pt_list:
                        prof = meas_dict.get(win_i)
                        if prof is not None and len(prof) > 0:
                            max_len = max(max_len, len(prof))

                if max_len == 0:
                    continue

                if is_complex:
                    reals = np.zeros((n, measurements_per_point, max_len), dtype=np.float64)
                    imags = np.zeros((n, measurements_per_point, max_len), dtype=np.float64)
                    for pt_idx, pt_list in enumerate(self._buf_profiles):
                        for meas_idx, meas_dict in enumerate(pt_list):
                            prof = meas_dict.get(win_i)
                            if prof is not None and len(prof) > 0:
                                length = min(len(prof), max_len)
                                reals[pt_idx, meas_idx, :length] = np.real(
                                    prof[:length]
                                )
                                imags[pt_idx, meas_idx, :length] = np.imag(
                                    prof[:length]
                                )
                    save_dict[f"profile_real_w{win_i}"] = reals
                    save_dict[f"profile_imag_w{win_i}"] = imags
                else:
                    mods = np.zeros((n, measurements_per_point, max_len), dtype=np.float64)
                    for pt_idx, pt_list in enumerate(self._buf_profiles):
                        for meas_idx, meas_dict in enumerate(pt_list):
                            prof = meas_dict.get(win_i)
                            if prof is not None and len(prof) > 0:
                                length = min(len(prof), max_len)
                                mods[pt_idx, meas_idx, :length] = np.abs(
                                    prof[:length]
                                )
                    save_dict[f"profile_mod_w{win_i}"] = mods

                if self._profile_depth_axes_m and win_i in self._profile_depth_axes_m:
                    save_dict[f"profile_depth_m_w{win_i}"] = np.asarray(
                        self._profile_depth_axes_m[win_i], dtype=np.float64
                    )

        # Ventanas (Schema 6.0: siempre presentes, ≥1 entrada)
        cm = self._extract_common_metadata(self._metadata)
        save_dict["win_depth_min_m"] = np.array(cm["win_depth_min_m"], dtype=np.float64)
        save_dict["win_depth_max_m"] = np.array(cm["win_depth_max_m"], dtype=np.float64)

        # Metadata — SIEMPRE completa, independiente de save_cfg
        save_dict.update(
            self._build_metadata_dict(self._metadata, n, aborted)
        )

        os.makedirs(os.path.dirname(self._filepath) or ".", exist_ok=True)
        np.savez(self._filepath, **save_dict)

        status = "ABORTADO" if aborted else "completo"
        logger.info("NPZ guardado (%s): %s (%d puntos)", status, self._filepath, n)
        path = self._filepath
        self._filepath = None
        return path

    # ── HDF5 — escritura incremental ──────────────────────────────────────────

    def _open_h5(self, filepath, planned_points, n_pixels, metadata, save_cfg):
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py no instalado. Ejecutar: pip install h5py")

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self._h5file = h5py.File(filepath, "w")
        self._h5_profiles_created = False
        self._h5_written_points = 0
        f = self._h5file
        m = self._measurements_per_point
        n_active = self._n_windows

        # maxshape=(None, ...) permite recortar a written_points al cerrar
        # un scan abortado: el archivo final contiene solo puntos reales.
        f.create_dataset("x_mm", shape=(planned_points,), maxshape=(None,), dtype=np.float64)
        f.create_dataset("y_mm", shape=(planned_points,), maxshape=(None,), dtype=np.float64)
        f.create_dataset("z_mm", shape=(planned_points,), maxshape=(None,), dtype=np.float64)

        # Picos dominantes — solo si el usuario los pidió
        if save_cfg.save_peaks:
            for name in ("depth_m", "amplitude"):
                f.create_dataset(
                    name,
                    shape=(planned_points, m, n_active),
                    maxshape=(None, m, n_active),
                    dtype=np.float64,
                    fillvalue=np.nan,
                )

        if save_cfg.save_spectra and n_pixels > 0:
            f.create_dataset(
                "spectra",
                shape=(planned_points, m, n_pixels),
                maxshape=(None, m, n_pixels),
                dtype=np.float64,
                compression="gzip",
                compression_opts=4,
                chunks=(max(1, min(64, planned_points)), m, n_pixels),
            )

        # Perfiles: creación diferida en el primer write_point, usando
        # los largos REALES de cada ventana (cada una tiene su n_z).

        self._write_h5_metadata(f, metadata, planned_points, n_pixels)
        logger.info("HDF5 abierto: %s (%d puntos, M=%d)", filepath, planned_points, m)

    def _ensure_h5_profile_datasets(self, profiles, profile_depth_axes_m):
        """
        Crear datasets de perfiles en el primer punto, dimensionados
        con el largo real del perfil de cada ventana activa. Las
        ventanas cuyo perfil llega vacío no generan dataset (el largo
        es determinístico por configuración: vacío en el primer punto
        implica vacío en todo el scan).
        """
        f = self._h5file
        m = self._measurements_per_point
        planned_points = self._planned_points
        is_complex = self._save_cfg.save_complex
        first_meas = profiles[0] if profiles else {}

        for win_i in range(self._n_windows):
            prof = first_meas.get(win_i)
            n_z_w = len(prof) if prof is not None else 0
            if n_z_w == 0:
                continue

            if profile_depth_axes_m and win_i in profile_depth_axes_m:
                f.create_dataset(
                    f"profile_depth_m_w{win_i}",
                    data=np.asarray(profile_depth_axes_m[win_i], dtype=np.float64),
                )

            names = (
                (f"profile_real_w{win_i}", f"profile_imag_w{win_i}")
                if is_complex
                else (f"profile_mod_w{win_i}",)
            )
            for name in names:
                f.create_dataset(
                    name,
                    shape=(planned_points, m, n_z_w),
                    maxshape=(None, m, n_z_w),
                    dtype=np.float64,
                    compression="gzip",
                    compression_opts=4,
                    chunks=(max(1, min(32, planned_points)), m, n_z_w),
                )
        self._h5_profiles_created = True

    def _write_point_h5(
        self,
        idx,
        x_mm,
        y_mm,
        z_mechanical_mm,
        spectra,
        depth_m,
        amplitude,
        profiles,
        profile_depth_axes_m,
    ):
        f = self._h5file
        f["x_mm"][idx] = x_mm
        f["y_mm"][idx] = y_mm
        f["z_mm"][idx] = z_mechanical_mm

        if self._save_cfg.save_peaks and depth_m is not None:
            f["depth_m"][idx] = depth_m
            f["amplitude"][idx] = amplitude

        if self._save_cfg.save_spectra and spectra is not None and "spectra" in f:
            f["spectra"][idx] = spectra

        if self._save_cfg.save_profile and profiles is not None:
            if not self._h5_profiles_created:
                self._ensure_h5_profile_datasets(profiles, profile_depth_axes_m)

            is_complex = self._save_cfg.save_complex
            for meas_idx, meas_dict in enumerate(profiles):
                for win_id, prof in meas_dict.items():
                    if is_complex:
                        ds_name = f"profile_real_w{win_id}"
                    else:
                        ds_name = f"profile_mod_w{win_id}"
                    if ds_name not in f:
                        continue
                    n_z_w = f[ds_name].shape[2]
                    length = min(len(prof), n_z_w)
                    if length == 0:
                        continue
                    if is_complex:
                        f[ds_name][idx, meas_idx, :length] = prof[:length].real
                        f[f"profile_imag_w{win_id}"][idx, meas_idx, :length] = prof[
                            :length
                        ].imag
                    else:
                        f[ds_name][idx, meas_idx, :length] = np.abs(prof[:length])

        self._h5_written_points = max(self._h5_written_points, idx + 1)

    def _close_h5(self, aborted: bool) -> Optional[str]:
        if self._h5file is None:
            return None
        f = self._h5file
        written_points = self._h5_written_points

        # Recortar todos los datasets de datos a los puntos realmente
        # escritos: un scan abortado no contiene cola pre-alocada.
        if written_points < self._planned_points:
            for name, ds in f.items():
                if hasattr(ds, "maxshape") and ds.maxshape and ds.maxshape[0] is None:
                    ds.resize(written_points, axis=0)

        f.attrs["aborted"] = aborted
        f.attrs["end_time"] = (
            self._end_time.isoformat() if self._end_time else ""
        )
        f.attrs["acquired_points"] = self._metadata.get(
            "acquired_points", written_points
        )
        f.flush()
        f.close()
        self._h5file = None
        self._h5_profiles_created = False
        status = "ABORTADO" if aborted else "completo"
        logger.info("HDF5 cerrado (%s): %s", status, self._filepath)
        path = self._filepath
        self._filepath = None
        return path

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _rename_aborted(filepath: str) -> str:
        """
        Renombrar archivo para indicar que el scan fue abortado.
        Inserta '_ABORTADO' antes de la extensión.

        Retorna el nuevo path (o el original si el rename falla).
        """
        base, ext = os.path.splitext(filepath)
        aborted_filepath = f"{base}_ABORTADO{ext}"
        candidate = aborted_filepath
        suffix = 1
        try:
            while os.path.exists(candidate):
                candidate = f"{base}_ABORTADO_{suffix}{ext}"
                suffix += 1
            os.rename(filepath, candidate)
            logger.info("Archivo renombrado: %s → %s", filepath, candidate)
            return candidate
        except OSError as e:
            logger.warning("No se pudo renombrar archivo abortado: %s", e)
            return filepath

    def _extract_common_metadata(self, metadata):
        """
        Extraer campos del Schema 6.0 en un dict canónico.
        Fuente única de verdad para NPZ y HDF5.

        Retorna dict con tipos Python nativos (str, int, float, bool).
        Cada formato convierte a su representación final (np.* para NPZ,
        attrs para HDF5).
        """
        win_cfg = metadata.get("windows", [])
        optics = metadata.get("optics", {})
        tolerance_mm = metadata.get("position_tolerance_mm")
        if tolerance_mm is None:
            tolerance_um = metadata.get("position_tolerance_um")
            tolerance_mm = (
                float(tolerance_um) * 1e-3
                if tolerance_um is not None
                else DEFAULT_POSITION_TOLERANCE_MM
            )
        tolerance_mm = float(tolerance_mm)

        d = {
            # Identificación
            "schema_version": self.schema_version,
            "software_version": str(metadata.get("software_version", "")),
            "sample_name": str(metadata.get("sample_name", "")),
            # Hardware
            "spectrometer_model": str(metadata.get("spectrometer_model", "")),
            "spectrometer_serial": str(metadata.get("spectrometer_serial", "")),
            # Adquisición
            "exposure_ms": float(metadata.get("exposure_ms", 0)),
            "dark_enabled": bool(metadata.get("dark_enabled", False)),
            "nonlinearity_enabled": bool(metadata.get("nonlinearity_enabled", False)),
            # Barrido
            "scan_mode": str(metadata.get("scan_mode", "snake")),
            "axis_order": str(metadata.get("axis_order", "XYZ")),
            "position_tolerance_mm": tolerance_mm,
            "position_tolerance_um": tolerance_mm * 1e3,
            # Ventanas (siempre ≥1 entrada; modo global = ventana única)
            "win_depth_min_m": [float(w["depth_min_m"]) for w in win_cfg],
            "win_depth_max_m": [float(w["depth_max_m"]) for w in win_cfg],
        }

        # Óptica (solo inputs con valor)
        for key in _OPTICS_KEYS:
            val = optics.get(key)
            if val is not None:
                d[f"optics_{key}"] = float(val)

        return d

    def _build_metadata_dict(self, metadata, n_acquired, aborted):
        """Metadata para NPZ: convierte a tipos numpy con keys UPPER_CASE."""
        cm = self._extract_common_metadata(metadata)

        d = {
            # Identificación
            "schema_version": cm["schema_version"],
            "software_version": cm["software_version"],
            "sample_name": cm["sample_name"],
            # Hardware
            "spectrometer_model": cm["spectrometer_model"],
            "spectrometer_serial": cm["spectrometer_serial"],
            # Adquisición
            "exposure_ms": np.float64(cm["exposure_ms"]),
            "dark": np.bool_(cm["dark_enabled"]),
            "nonlinearity": np.bool_(cm["nonlinearity_enabled"]),
            # Barrido
            "scan_mode": cm["scan_mode"],
            "axis_order": cm["axis_order"],
            "position_tolerance_mm": np.float64(cm["position_tolerance_mm"]),
            "position_tolerance_um": np.float64(cm["position_tolerance_um"]),
            # Temporal
            "start_time": self._start_time.isoformat() if self._start_time else "",
            "end_time": self._end_time.isoformat() if self._end_time else "",
            # Estado
            "planned_points": np.int32(
                metadata.get("planned_points", n_acquired)
            ),
            "acquired_points": np.int32(
                metadata.get("acquired_points", n_acquired)
            ),
            "aborted": np.bool_(aborted),
        }

        # Óptica
        for key in _OPTICS_KEYS:
            cm_key = f"optics_{key}"
            if cm_key in cm:
                d[_PERSISTED_OPTICS_KEYS[key]] = np.float64(cm[cm_key])

        return d

    def _write_h5_metadata(self, f, metadata, planned_points, n_pixels):
        """Metadata para HDF5: escribe en attrs con keys lowercase."""
        cm = self._extract_common_metadata(metadata)
        attrs = f.attrs

        # Campos directos desde common metadata
        for key in (
            "schema_version", "software_version", "sample_name",
            "spectrometer_model", "spectrometer_serial",
            "exposure_ms",
            "scan_mode", "axis_order",
            "position_tolerance_mm", "position_tolerance_um",
        ):
            attrs[key] = cm[key]

        attrs["dark"] = cm["dark_enabled"]
        attrs["nonlinearity"] = cm["nonlinearity_enabled"]

        # Temporal: start_time se escribe al abrir (sobrevive a crash).
        # end_time se escribe al cerrar (_close_h5).
        attrs["start_time"] = (
            self._start_time.isoformat() if self._start_time else ""
        )

        # Estado
        attrs["planned_points"] = planned_points

        # Datasets fijos: wavelengths_nm y geometría de ventanas
        f.create_dataset("wavelengths_nm", data=self._wavelengths_nm)
        f.create_dataset("win_depth_min_m", data=np.array(cm["win_depth_min_m"]))
        f.create_dataset("win_depth_max_m", data=np.array(cm["win_depth_max_m"]))

        # Óptica
        for key in _OPTICS_KEYS:
            cm_key = f"optics_{key}"
            if cm_key in cm:
                attrs[_PERSISTED_OPTICS_KEYS[key]] = cm[cm_key]
