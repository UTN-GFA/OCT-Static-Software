# -*- coding: utf-8 -*-
"""
gui/scan_controller.py
ScanController — orquestador del barrido OCT.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Callable

import numpy as np

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanSummary:
    """
    Resumen del scan que el controller envía con scan_finished/aborted.
    La GUI NO consulta estado interno del controller — recibe esto.
    """

    acquired_points: int
    written_points: int
    planned_points: int
    duration_s: float
    aborted: bool

    @property
    def completion_pct(self) -> float:
        return 100.0 * self.acquired_points / max(self.planned_points, 1)


class ScanController(QObject):
    """
    Orquestador del scan. Owner de: buffers, saver, ScanWorker.

    Señales (la GUI se suscribe):
        point_ready(x_mm, y_mm, z_mechanical_mm, raw, result)
            → para visualización
        progress(acquired, total, eta_str, x_mm, y_mm, z_mechanical_mm)
            → para progress bar y label
        scan_finished(filepath)            → scan completo OK
        scan_aborted(filepath)             → scan abortado por usuario
        scan_error(msg)                    → error terminal del worker

    La GUI NO debe acceder directamente a los buffers (son internos).
    """

    point_ready = Signal(float, float, float, object, object)
    progress = Signal(
        int, int, str, float, float, float
    )  # acquired, total, eta, x_mm, y_mm, z_mechanical_mm
    scan_finished = Signal(str, object)  # filepath, ScanSummary
    scan_aborted = Signal(str, object)  # filepath, ScanSummary
    scan_error = Signal(str)  # error_msg

    def __init__(self, acq_engine, proc_engine, saver, scan_worker_factory: Callable):
        """
        Parámetros
        ----------
        acq_engine : AcquisitionReader
        proc_engine : ProcessingEngine
        saver : OCTDataSaver
        scan_worker_factory : callable
            Factory que construye un ScanWorker.
            Signature: (mot, acq, proc, scan_cfg, measurements_per_point) -> QThread
            Se pasa como factory para no importar ScanWorker desde aquí
            (evita ciclo: gui.main_gui define ScanWorker, importa este módulo).

        Nota: mot NO se pasa al constructor. Se pasa a start() para que
        cada ejecución sea autocontenida (permite reconectar motores
        entre scans sin recrear el controller).
        """
        super().__init__()
        self._acq = acq_engine
        self._proc = proc_engine
        self._saver = saver
        self._worker_factory = scan_worker_factory
        self._worker = None

        # Buffers del scan actual
        self._scan_x_mm: list = []
        self._scan_y_mm: list = []
        self._scan_z_mechanical_mm: list = []

        # Estado del scan
        self._save_cfg = None
        self._write_error: Optional[str] = None  # error de escritura pendiente
        self._win_indices: list = []  # mapeo columna → índice de ventana original
        self._planned_points = 0
        self._acquired_points = 0
        self._written_points = 0
        self._start_time: Optional[datetime] = None
        self._metadata: Optional[dict] = None

    # ═══════════════════════════════════════════════════════════
    # API pública
    # ═══════════════════════════════════════════════════════════

    def start(
        self,
        mot,
        scan_cfg,
        proc_cfg,
        save_cfg,
        metadata: dict,
        filepath: str,
        file_format: str,
        n_pixels: int,
        position_tolerance_mm: float,
        wavelengths_nm: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Iniciar el scan.

        Parámetros
        ----------
        mot : MotionControllerInterface
            Motor a usar. Se pasa en cada start() para permitir
            reconexión entre scans sin recrear el controller.
        scan_cfg : ScanConfig
        proc_cfg : ProcessingConfig
        save_cfg : SaveConfig
        metadata : dict
            Metadata inicial para open_scan. La GUI la construye.
        filepath : str
            Path del archivo de salida.
        file_format : str
            "npz" o "h5".
        n_pixels : int
            Cantidad de píxeles del espectrómetro.
        position_tolerance_mm : float
            Error máximo permitido entre objetivo y posición alcanzada.
        wavelengths_nm : np.ndarray
            Calibración píxel→λ (nm). Obligatorio en Schema 6.0.

        Retorna
        -------
        True si el scan arrancó, False si falló al abrir el archivo.
        En caso False, el controller emite scan_error.
        """
        if self._worker is not None and self._worker.isRunning():
            logger.warning("start() ignorado: scan ya en curso")
            return False

        try:
            measurements_per_point = int(proc_cfg.measurements_per_point)
        except (AttributeError, TypeError, ValueError) as exc:
            message = f"measurements_per_point inválido: {exc}"
            logger.error(message)
            self.scan_error.emit(message)
            return False
        if measurements_per_point < 1:
            message = (
                "measurements_per_point inválido: debe ser mayor o igual a 1"
            )
            logger.error(message)
            self.scan_error.emit(message)
            return False

        # Reset de buffers y estado
        self._scan_x_mm.clear()
        self._scan_y_mm.clear()
        self._scan_z_mechanical_mm.clear()

        self._save_cfg = save_cfg
        self._planned_points = scan_cfg.planned_points
        self._acquired_points = 0
        self._written_points = 0
        self._start_time = datetime.now()
        self._write_error = None
        self._metadata = metadata
        metadata["planned_points"] = self._planned_points
        metadata["acquired_points"] = 0

        # Mapeo columna → ventana original (interno, para traducir
        # keys del engine a columnas secuenciales de depth/amplitude).
        active = proc_cfg.active_windows
        self._win_indices = [w.index for w in active] if active else [0]

        # Abrir archivo
        try:
            self._saver.open_scan(
                filepath=filepath,
                planned_points=self._planned_points,
                n_pixels=n_pixels,
                metadata=metadata,
                save_cfg=save_cfg,
                file_format=file_format,
                measurements_per_point=measurements_per_point,
                wavelengths_nm=wavelengths_nm,
            )
        except Exception as e:
            logger.error("open_scan falló: %s", e, exc_info=True)
            self.scan_error.emit(str(e))
            return False

        # Crear y arrancar worker con el motor del scan
        try:
            self._worker = self._worker_factory(
                mot,
                self._acq,
                self._proc,
                scan_cfg,
                measurements_per_point,
                position_tolerance_mm,
            )
            self._worker.point_ready.connect(self._on_point)
            self._worker.finished.connect(self._on_finished)
            self._worker.error_occurred.connect(self._on_error)
            self._worker.aborted.connect(self._on_aborted)
            self._worker.start()
        except Exception as e:
            logger.error("No se pudo crear o arrancar el worker: %s", e, exc_info=True)
            self._update_final_metadata_counts()
            try:
                self._saver.close_scan(aborted=True)
            except Exception as close_error:
                logger.error("Error cerrando archivo tras fallo del worker: %s", close_error, exc_info=True)
            self._worker = None
            self.scan_error.emit(str(e))
            return False
        logger.info("Scan iniciado — %d puntos", self._planned_points)
        return True

    def abort(self):
        """Solicitar abort del scan en curso."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.abort()

    def wait_finish(self, timeout_ms: int = 5000) -> bool:
        """
        Esperar a que el worker termine (bloqueante).
        Usado en closeEvent de la GUI.
        Retorna True si terminó, False si se cumplió el timeout.
        """
        if self._worker is None:
            return True
        return self._worker.wait(timeout_ms)

    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    # ═══════════════════════════════════════════════════════════
    # Slots del ScanWorker
    # ═══════════════════════════════════════════════════════════

    def _on_point(self, x_mm, y_mm, z_mechanical_mm, raw, result):
        """Worker entregó un punto: actualizar buffers, escribir, emitir."""
        try:
            # Pico dominante de TODAS las M mediciones.
            # Shapes: (M, N_active) — solo ventanas activas, sin relleno
            # estructural. NaN únicamente donde una medición no produjo
            # pico (ventana sin muestras), que sí es dato real.
            depth_m_by_measurement = None
            amplitude_by_measurement = None
            if self._save_cfg.save_peaks:
                m = result.measurements_per_point
                n_w = len(self._win_indices)
                depth_m_by_measurement = np.full((m, n_w), np.nan)
                amplitude_by_measurement = np.full((m, n_w), np.nan)
                for meas_idx, pk_dict in enumerate(result.peaks):
                    for col, win_id in enumerate(self._win_indices):
                        pk = pk_dict.get(win_id)
                        if pk is not None and pk.depth_m is not None:
                            depth_m_by_measurement[meas_idx, col] = pk.depth_m
                            amplitude_by_measurement[meas_idx, col] = pk.amplitude

            idx = self._acquired_points
            self._acquired_points += 1
            self._scan_x_mm.append(x_mm)
            self._scan_y_mm.append(y_mm)
            self._scan_z_mechanical_mm.append(z_mechanical_mm)
        except Exception as e:
            logger.error("Error construyendo datos del punto (%s,%s,%s): %s",
                         x_mm, y_mm, z_mechanical_mm, e, exc_info=True)
            self._write_error = str(e)
            if self._worker is not None:
                self._worker.abort()
            return

        # Escritura a disco — fuera del catch-all. Si falla, el scan
        # se detiene vía abort y la notificación sale por _on_aborted.
        if self._saver.is_open():
            try:
                spectra = self._build_spectrum_for_save(raw)
                profiles, profile_depth_axes_m = self._build_profiles_for_save(result)

                self._saver.write_point(
                    idx=idx,
                    x_mm=x_mm,
                    y_mm=y_mm,
                    z_mechanical_mm=z_mechanical_mm,
                    spectra=spectra,
                    depth_m=depth_m_by_measurement,
                    amplitude=amplitude_by_measurement,
                    profiles=profiles,
                    profile_depth_axes_m=profile_depth_axes_m,
                )
            except Exception as e:
                logger.error("Error de escritura en punto %d: %s",
                             idx, e, exc_info=True)
                self._write_error = str(e)
                if self._worker is not None:
                    self._worker.abort()
                return

        self._written_points += 1

        # Solo si la escritura fue exitosa:
        self.point_ready.emit(x_mm, y_mm, z_mechanical_mm, raw, result)

        # Progreso + ETA + última posición (para la GUI)
        eta_str = self._compute_eta()
        self.progress.emit(
            self._acquired_points,
            self._planned_points,
            eta_str,
            x_mm,
            y_mm,
            z_mechanical_mm,
        )

    def _build_summary(self, aborted: bool) -> ScanSummary:
        """Construye el resumen que se emite al terminar el scan."""
        elapsed = (
            (datetime.now() - self._start_time).total_seconds()
            if self._start_time
            else 0.0
        )
        return ScanSummary(
            acquired_points=self._acquired_points,
            written_points=self._written_points,
            planned_points=self._planned_points,
            duration_s=elapsed,
            aborted=aborted,
        )

    def _on_finished(self):
        self._update_final_metadata_counts()
        try:
            path = self._saver.close_scan(aborted=False)
        except Exception as e:
            logger.error("Error cerrando archivo: %s", e, exc_info=True)
            path = ""
        summary = self._build_summary(aborted=False)
        self._worker = None
        self.scan_finished.emit(path or "", summary)

    def _on_error(self, error_msg: str):
        self._update_final_metadata_counts()
        try:
            self._saver.close_scan(aborted=True)
        except Exception as e:
            logger.error("Error cerrando archivo: %s", e, exc_info=True)
        self._worker = None
        self.scan_error.emit(error_msg)

    def _on_aborted(self):
        self._update_final_metadata_counts()
        try:
            path = self._saver.close_scan(aborted=True)
        except Exception as e:
            logger.error("Error cerrando archivo: %s", e, exc_info=True)
            path = ""

        # Si el abort fue disparado por un error de escritura
        # (_on_point detectó fallo en write_point), la notificación
        # sale como scan_error — no como scan_aborted.
        if self._write_error:
            msg = self._write_error
            self._write_error = None
            self._worker = None
            self.scan_error.emit(f"Error de escritura: {msg}")
        else:
            summary = self._build_summary(aborted=True)
            self._worker = None
            self.scan_aborted.emit(path or "", summary)

    # ═══════════════════════════════════════════════════════════
    # Internals
    # ═══════════════════════════════════════════════════════════

    def _update_final_metadata_counts(self):
        """Publicar los conteos reales antes de cerrar el artefacto."""
        if self._metadata is not None:
            self._metadata["planned_points"] = self._planned_points
            self._metadata["acquired_points"] = self._acquired_points

    def _build_spectrum_for_save(self, raw):
        """
        Devuelve los espectros crudos a guardar para este punto.
        Shape: (M, N_pixels) — una fila por cada medición.
        """
        if not self._save_cfg.save_spectra:
            return None
        return raw.spectra  # (M, N_pixels)

    def _build_profiles_for_save(self, result):
        """
        Devuelve los perfiles axiales a guardar con keys secuenciales.

        Retorna
        -------
        tuple[profiles, profile_depth_axes_m]
            profiles es una lista de M dicts; profile_depth_axes_m
            contiene el eje depth_axis_m de cada ventana. None si
            save_profile está deshabilitado.
        """
        if not self._save_cfg.save_profile:
            return None, None

        is_complex = self._save_cfg.profile_mode == "complex"
        out = []
        profile_depth_axes_m = {}
        for wr_dict in result.window_results:
            renumbered = {}
            for col, orig_id in enumerate(self._win_indices):
                wr = wr_dict.get(orig_id)
                if wr is not None:
                    profile_depth_axes_m.setdefault(col, wr.depth_axis_m)
                    renumbered[col] = wr.profile if is_complex else np.abs(wr.profile)
            out.append(renumbered)
        return out, profile_depth_axes_m

    def _compute_eta(self) -> str:
        elapsed = (datetime.now() - self._start_time).total_seconds()
        if elapsed <= 0:
            return "calculando..."
        rate = self._acquired_points / elapsed
        remaining = (self._planned_points - self._acquired_points) / max(rate, 1e-6)
        mins, secs = divmod(int(remaining), 60)
        return f"ETA {mins}m{secs:02d}s"
