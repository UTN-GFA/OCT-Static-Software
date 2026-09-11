# -*- coding: utf-8 -*-
"""
gui/main_gui.py
GUI principal OCT
"""

import numpy as np
from datetime import datetime
from typing import Optional
import os
import logging
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget,
    QMainWindow,
    QPushButton,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QMessageBox,
    QTabWidget,
    QComboBox,
    QSplitter,
    QProgressBar,
    QLineEdit,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap
import pyqtgraph as pg

# ── Capa de procesamiento ─────────────────────────────────────
from processing import (
    ProcessingEngine,
    ProcessingConfig,
    WindowConfig,
    ProcessingResult,
)
from processing.optics import calculate_optics
from processing.peaks import analyze_profile_peaks

# ── Capa de adquisición ───────────────────────────────────────
from acquisition import AcquisitionReader, ScanConfig

# ── Hardware vía factory ──────────────────────────────────────
from hardware import (
    create_spectrometer,
    create_motion_controller,
    list_spectrometers,
    list_motion_controllers,
)
from hardware.spectrometer_mock import Reflector

# ── Guardado ──────────────────────────────────────────────────
from storage.saver import OCTDataSaver, SaveConfig, generate_filename

# ── Constantes compartidas ────────────────────────────────────
from constants import (
    MAX_WINDOWS,
    AXIS_NAMES,
    DEFAULT_EXPOSURE_MS,
    DEFAULT_FIBER_DIAMETER_UM,
    DEFAULT_WAVELENGTH_NM,
    DEFAULT_COLLIMATOR_FOCAL_LENGTH_MM,
    DEFAULT_OBJECTIVE_FOCAL_LENGTH_MM,
    DEFAULT_N_PIXELS,
    DEFAULT_SETTLING_TIME_S,
    DEFAULT_POSITION_TOLERANCE_UM,
    MIN_POSITION_TOLERANCE_UM,
    DEFAULT_SPECTROMETER_KIND,
    DEFAULT_MOTION_CONTROLLER_KIND,
    DEFAULT_MOTOR_PORT,
    MOCK_MOTION_SPEED_MM_S,
    MOCK_DARK_COUNTS,
    SOFTWARE_NAME,
    SOFTWARE_VERSION,
)

# ── Orquestador del scan ──────────────────────────────────────
from gui.scan_controller import ScanController

# ── Workers y estados ─────────────────────────────────────────
from gui.workers import SystemState, MonitorWorker, ScanWorker, MoveWorker

logger = logging.getLogger(__name__)

# ================= CONFIG =================
# Amarillo reservado exclusivamente para el espectro y el perfil global.
SPECTRUM_COLOR = "y"
GLOBAL_PROFILE_COLOR = SPECTRUM_COLOR
# Las ventanas activas usan colores diferenciados, nunca amarillo.
WINDOW_COLORS = ["c", "m", "g", "r", "b"]
# =========================================


# ============================================================
# GUI PRINCIPAL V6.0
# ============================================================
class OCTGUI(QMainWindow):

    def __init__(self, use_mock: bool = False):
        super().__init__()

        # ── Estado del sistema ────────────────────────────────────
        self._state = SystemState.IDLE

        # ── Hardware ──────────────────────────────────────────────
        self._use_mock = use_mock
        self._spectrometer_kind = "mock" if use_mock else DEFAULT_SPECTROMETER_KIND
        self._motion_controller_kind = "mock" if use_mock else DEFAULT_MOTION_CONTROLLER_KIND
        self.spec = self._init_spectrometer(use_mock, kind=self._spectrometer_kind)
        self._applied_exposure_ms = DEFAULT_EXPOSURE_MS
        self.mot = self._init_motors(
            use_mock=use_mock,
            kind=self._motion_controller_kind,
        )
        # Estado de habilitación por eje. El hardware no expone una
        # consulta uniforme de este estado, así que la GUI lo mantiene
        # sincronizado con los botones Enable/Disable.
        self._motor_enabled = {1: False, 2: False, 3: False}

        # Sincronización periódica de posición real de motores.
        # Se inicia después de construir la UI para que los widgets existan.
        self._motor_position_timer = QTimer(self)
        self._motor_position_timer.setInterval(200)
        self._motor_position_timer.timeout.connect(self._sync_motor_positions)

        # ── Processing Engine ─────────────────────────────────────
        self.engine = ProcessingEngine(ProcessingConfig())
        logger.info("ProcessingEngine inicializado")

        # ── Acquisition Engine ────────────────────────────────────
        self.acq = AcquisitionReader(self.spec, ScanConfig())
        logger.info("AcquisitionReader inicializado")

        # ── Saver ─────────────────────────────────────────────────
        self.saver = OCTDataSaver()

        # ── ScanController — dueño del estado del scan ────────────
        # Los buffers, progreso, ETA y el saver.write_point
        # viven en el controller. La GUI solo arma configs y grafica.
        # mot NO se pasa al constructor: se pasa a start() para permitir
        # reconexión entre scans sin recrear el controller.
        self._scan_controller = ScanController(
            acq_engine=self.acq,
            proc_engine=self.engine,
            saver=self.saver,
            scan_worker_factory=lambda mot, acq, proc, cfg, n, tolerance_mm: ScanWorker(
                mot, acq, proc, cfg, n, tolerance_mm
            ),
        )
        self._scan_controller.point_ready.connect(self._on_scan_point)
        self._scan_controller.progress.connect(self._on_scan_progress)
        self._scan_controller.scan_finished.connect(self._on_scan_finished)
        self._scan_controller.scan_aborted.connect(self._on_scan_aborted)
        self._scan_controller.scan_error.connect(self._on_scan_error)

        # ── n_pixels dinámico (fallback definido en constants.py) ───
        # Se cachea al leer el primer espectro (monitor o scan).
        # Mientras tanto, se intenta inferir desde el espectrómetro.
        self._n_pixels_cached: Optional[int] = None
        self._wavelengths_cached_nm: Optional[np.ndarray] = None
        self._axial_range_mm: Optional[float] = None

        # ── Workers ───────────────────────────────────────────────
        self._monitor_worker = None
        self._move_worker = None
        # Botones de movimiento manual (se llenan en _build_tab_drivers).
        # Se deshabilitan mientras un MoveWorker está en curso.
        self._manual_move_buttons = []

        # ── Último ProcessingResult graficado ─────────────────────
        # Lo usa la herramienta "Analizar picos" (inspección bajo
        # demanda). No forma parte del pipeline ni del guardado.
        self._last_result: Optional[ProcessingResult] = None

        # ── Metadata congelada durante scan ───────────────────────
        self._frozen_gui_params = None

        # ── UI ────────────────────────────────────────────────────
        self._setup_ui()
        self._refresh_engine()
        self._motor_position_timer.start()

    # ============================================================
    # Máquina de estados
    # ============================================================
    def _set_state(self, state: SystemState):
        self._state = state
        is_idle = state == SystemState.IDLE
        is_monitoring = state == SystemState.MONITORING
        is_scanning = state == SystemState.SCANNING

        if hasattr(self, "btn_start_monitor"):
            self.btn_start_monitor.setEnabled(is_idle or state == SystemState.ERROR)
            self.btn_stop_monitor.setEnabled(is_monitoring)

        if hasattr(self, "btn_run_scan"):
            self._update_run_scan_button()
            self.btn_abort_scan.setEnabled(is_scanning)

        # ── Bloqueo de UI durante SCANNING ────────────────────────
        # Tab Drivers: bloqueado completo durante scan.
        # Tab Monitor: la configuración de ventanas y la visualización
        #   quedan accesibles durante un monitor, pero se bloquean durante
        #   un barrido si corresponde.
        # Tab Barrido: todo bloqueado EXCEPTO Abort y "Pausar gráficos".
        if hasattr(self, "_tabs"):
            # Tab 0 = Drivers → bloqueado completo durante scan
            self._tabs.setTabEnabled(0, not is_scanning)

            # Reconexión de hardware: bloqueada durante scan, monitor,
            # y modo mock (no tiene sentido reconectar simulación).
            move_active = (
                self._move_worker is not None
                and self._move_worker.isRunning()
            )
            not_mock_and_available = (
                is_idle or state == SystemState.ERROR
            ) and not self._use_mock
            for attr in (
                "btn_motor_reconnect",
                "combo_port",
                "combo_motion_controller",
            ):
                w = getattr(self, attr, None)
                if w is not None:
                    w.setEnabled(not_mock_and_available and not move_active)
            for attr in ("btn_spec_reconnect", "combo_spectrometer"):
                w = getattr(self, attr, None)
                if w is not None:
                    w.setEnabled(not_mock_and_available)

            # Movimiento manual: además del bloqueo por worker, queda
            # bloqueado mientras corre un barrido.
            self._set_manual_buttons_enabled(
                not is_scanning and not move_active
            )

            # Drivers → widgets de adquisición
            for attr_name in (
                "spin_exposure_ms",
                "chk_dark",
                "chk_nonlinearity",
                # La configuración óptica forma parte de la metadata
                # congelada al iniciar el scan; no debe editarse durante
                # SCANNING.
                "fiber_diameter_spinbox",
                "wavelength_central_spinbox",
                "collimator_focal_length_spinbox",
                "objective_focal_length_spinbox",
            ):
                w = getattr(self, attr_name, None)
                if w is not None:
                    w.setEnabled(not is_scanning)

            # Tab 2 = Barrido → widgets de configuración
            for attr_name in (
                "edit_sample_name",
                "spin_measurements_per_point",
                "combo_scan_mode",
                "combo_axis_order",
                "spin_settling_time_s",
                "save_format_combo_box",
                "chk_save_spectra",
                "chk_save_peaks",
                "chk_save_profile",
            ):
                w = getattr(self, attr_name, None)
                if w is not None:
                    w.setEnabled(not is_scanning)

            # El modo del perfil axial depende de dos condiciones:
            # el guardado del perfil debe estar seleccionado y no debe
            # haber un barrido activo. No habilitarlo sólo por salir de
            # SCANNING, porque el checkbox puede seguir desmarcado.
            self._sync_profile_controls()

            # Spinboxes y checkboxes de ejes
            if hasattr(self, "scan_spinboxes"):
                for axis in self.scan_spinboxes:
                    for sp in self.scan_spinboxes[axis]:
                        sp.setEnabled(not is_scanning)
            self._update_scan_axis_checks()
            self._apply_hardware_capabilities()

            # Ventanas Profundidad y resolución global (tab Monitor)
            if hasattr(self, "window_enabled_checkboxes"):
                for i in range(len(self.window_enabled_checkboxes)):
                    self.window_enabled_checkboxes[i].setEnabled(not is_scanning)
                    self.window_depth_min_spinboxes[i].setEnabled(not is_scanning)
                    self.window_depth_max_spinboxes[i].setEnabled(not is_scanning)
                    self.window_resolution_spinboxes[i].setEnabled(not is_scanning)
            for attr_name in ("spin_global_depth_min_mm", "spin_global_resolution_um"):
                w = getattr(self, attr_name, None)
                if w is not None:
                    w.setEnabled(not is_scanning)

        state_labels = {
            SystemState.IDLE: ("IDLE", "neutral"),
            SystemState.MONITORING: ("MONITOR", "ok"),
            SystemState.SCANNING: ("SCANNING", "warning"),
            SystemState.ERROR: ("ERROR", "error"),
        }
        label, kind = state_labels[state]
        if hasattr(self, "lbl_system_state"):
            self._set_status_badge(self.lbl_system_state, f"Estado: {label}", kind)

        logger.info("Estado: %s", state.name)

    def _update_run_scan_button(self):
        """Habilitar Ejecutar sólo si el sistema está libre de movimiento manual."""
        if not hasattr(self, "btn_run_scan"):
            return
        move_active = (
            self._move_worker is not None
            and self._move_worker.isRunning()
        )
        can_run = self._state in (SystemState.IDLE, SystemState.MONITORING)
        self.btn_run_scan.setEnabled(can_run and not move_active)

    # ============================================================
    # Hardware
    # ============================================================
    def _init_spectrometer(self, use_mock: bool, kind: Optional[str] = None):
        kind = kind or ("mock" if use_mock else DEFAULT_SPECTROMETER_KIND)
        if use_mock:
            kind = "mock"
            spec = create_spectrometer(
                kind,
                dark_counts=MOCK_DARK_COUNTS,
                reflectors=[
                    Reflector(depth_m=0.5e-3, amplitude=1.0),
                    Reflector(depth_m=1.2e-3, amplitude=0.6),
                ],
            )
            spec.connect()
            logger.info("MockSpectrometer conectado")
            return spec

        try:
            spec = create_spectrometer(kind)
            if spec.connect():
                logger.info("Espectrómetro %s conectado", kind)
                return spec
        except Exception as e:
            logger.warning("Espectrómetro %s no disponible: %s", kind, e)
        return None

    def _init_motors(
        self,
        use_mock: bool = False,
        port: str = DEFAULT_MOTOR_PORT,
        kind: Optional[str] = None,
    ):
        kind = kind or ("mock" if use_mock else DEFAULT_MOTION_CONTROLLER_KIND)
        if use_mock:
            kind = "mock"
            mot = create_motion_controller(
                kind,
                motion_speed_mm_s=MOCK_MOTION_SPEED_MM_S,
            )
            mot.connect()
            logger.info("MockMotionController conectado")
            return mot

        try:
            mot = create_motion_controller(kind, port=port)
            if mot.connect():
                logger.info(
                    "Motores %s conectados en %s, ejes: %s",
                    kind,
                    port,
                    mot.available_axes,
                )
                return mot
        except Exception as e:
            logger.error("Motores %s no disponibles en %s: %s", kind, port, e)
        return None

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            blocked = combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(blocked)

    @staticmethod
    def _device_capabilities(device, defaults: dict) -> dict:
        """Leer capacidades sin romper drivers antiguos durante la transición."""
        if device is None:
            return dict(defaults)
        try:
            declared = device.capabilities()
        except (AttributeError, TypeError, RuntimeError):
            return dict(defaults)
        result = dict(defaults)
        if isinstance(declared, dict):
            result.update(declared)
        return result

    def _spectrometer_capabilities(self):
        caps = self._device_capabilities(
            self.spec,
            {
                "exposure_control": True,
                "dark_correction": True,
                "nonlinearity_correction": True,
                "wavelengths_nm": True,
                "reconnect": True,
            },
        )
        if self.spec is None:
            for key in (
                "exposure_control",
                "dark_correction",
                "nonlinearity_correction",
                "wavelengths_nm",
            ):
                caps[key] = False
        return caps

    def _motor_capabilities(self):
        caps = self._device_capabilities(
            self.mot,
            {
                "absolute_move": True,
                "relative_move": False,
                "position_readback": True,
                "stop_motion": True,
                "axis_enable": True,
                "homing": False,
                "velocity_control": False,
                "limits": False,
            },
        )
        if self.mot is None:
            return {key: False for key in caps}
        return caps

    def _apply_hardware_capabilities(self):
        """Reflejar capacidades del driver en los controles de la GUI."""
        state_allows_acquisition = self._state != SystemState.SCANNING
        spec_caps = self._spectrometer_capabilities()
        if hasattr(self, "spin_exposure_ms"):
            self.spin_exposure_ms.setEnabled(
                state_allows_acquisition
                and spec_caps["exposure_control"]
            )
        if hasattr(self, "chk_dark"):
            supported = spec_caps["dark_correction"]
            if not supported and self.chk_dark.isChecked():
                self.chk_dark.blockSignals(True)
                self.chk_dark.setChecked(False)
                self.chk_dark.blockSignals(False)
            self.chk_dark.setEnabled(supported and state_allows_acquisition)
            supported = spec_caps["nonlinearity_correction"]
            if not supported and self.chk_nonlinearity.isChecked():
                self.chk_nonlinearity.blockSignals(True)
                self.chk_nonlinearity.setChecked(False)
                self.chk_nonlinearity.blockSignals(False)
            self.chk_nonlinearity.setEnabled(
                supported and state_allows_acquisition
            )
        if hasattr(self, "btn_spec_reconnect"):
            self.btn_spec_reconnect.setEnabled(
                not self._use_mock
                and spec_caps["reconnect"]
                and self._state in (SystemState.IDLE, SystemState.ERROR)
            )

        motor_caps = self._motor_capabilities()
        if hasattr(self, "btn_motion_stop"):
            self.btn_motion_stop.setEnabled(
                motor_caps["stop_motion"]
                and self._move_worker is not None
                and self._move_worker.isRunning()
            )
        self._update_motor_axis_controls()
        self._update_scan_axis_checks()
        if hasattr(self, "_manual_move_buttons"):
            self._set_manual_buttons_enabled(
                self._state != SystemState.SCANNING
                and not (
                    self._move_worker is not None
                    and self._move_worker.isRunning()
                )
            )

    def _apply_spectrometer_settings(self):
        """Aplicar sólo los ajustes declarados por el espectrómetro."""
        if self.spec is None:
            return
        caps = self._spectrometer_capabilities()
        if caps["exposure_control"]:
            self.spec.set_exposure_ms(self.spin_exposure_ms.value())
            self._applied_exposure_ms = self.spin_exposure_ms.value()
        if caps["dark_correction"]:
            self.spec.set_dark_correction(self.chk_dark.isChecked())
        if caps["nonlinearity_correction"]:
            self.spec.set_nonlinearity_correction(
                self.chk_nonlinearity.isChecked()
            )

    def _on_spectrometer_selected(self, _index: int):
        if self._use_mock:
            return
        kind = self.combo_spectrometer.currentData()
        if not kind or kind == self._spectrometer_kind:
            return
        self._spectrometer_kind = kind
        self._reconnect_spectrometer()

    def _on_motion_controller_selected(self, _index: int):
        if self._use_mock:
            return
        kind = self.combo_motion_controller.currentData()
        if not kind or kind == self._motion_controller_kind:
            return
        self._motion_controller_kind = kind
        self._reconnect_motors()

    @property
    def available_axes(self) -> set:
        return self.mot.available_axes if self.mot else set()

    # ============================================================
    # Construcción de configuraciones — SIN lógica de procesamiento
    # ============================================================
    def _build_processing_config(self) -> ProcessingConfig:
        """
        Lee los widgets y delega la lógica a ProcessingConfig.from_gui_params().
        La GUI NO decide use_windows, NO calcula rangos z derivados.
        Pasa resolución en µm — from_gui_params calcula n_points.
        """
        windows = []
        if hasattr(self, "window_enabled_checkboxes"):
            for i in range(MAX_WINDOWS):
                res_um = self.window_resolution_spinboxes[i].value()
                depth_min_m = self.window_depth_min_spinboxes[i].value() * 1e-3
                depth_max_m = self.window_depth_max_spinboxes[i].value() * 1e-3
                depth_range_m = depth_max_m - depth_min_m
                n_pts = max(64, int(depth_range_m / (res_um * 1e-6))) if depth_range_m > 0 else 64
                windows.append(
                    WindowConfig(
                        index=i,
                        depth_min_m=depth_min_m,
                        depth_max_m=depth_max_m,
                        window_profile_samples=n_pts,
                        enabled=self.window_enabled_checkboxes[i].isChecked(),
                    )
                )

        n_k = self._get_n_pixels()

        depth_min_global_m = (
            self.spin_global_depth_min_mm.value() * 1e-3
            if hasattr(self, "spin_global_depth_min_mm")
            else 0.05e-3
        )
        depth_max_global_m = self._get_axial_range_m()
        if depth_max_global_m is None:
            raise ValueError(
                "Rango axial no disponible: se requiere una calibración espectral válida"
            )
        if depth_max_global_m <= depth_min_global_m:
            raise ValueError(
                "La profundidad mínima global debe ser menor que el rango axial"
            )

        global_res_um = (
            self.spin_global_resolution_um.value() if hasattr(self, "spin_global_resolution_um") else 1.0
        )
        depth_global_range_m = depth_max_global_m - depth_min_global_m
        global_profile_samples = max(64, int(depth_global_range_m / (global_res_um * 1e-6)))

        measurements_per_point = self.spin_measurements_per_point.value() if hasattr(self, "spin_measurements_per_point") else 1

        return ProcessingConfig.from_gui_params(
            windows=windows,
            depth_min_global_m=depth_min_global_m,
            depth_max_global_m=depth_max_global_m,
            n_k=n_k,
            global_profile_samples=global_profile_samples,
            measurements_per_point=measurements_per_point,
        )

    def _get_n_pixels(self) -> int:
        """
        Obtener n_pixels dinámicamente y cachear wavelengths.
        Prioridad:
          1) Valor cacheado del primer espectro leído.
          2) Lectura puntual del espectrómetro (try-read).
          3) Atributo n_pixels del espectrómetro (fallback sin wavelengths).
          4) Fallback 3648 — último recurso.
        """
        if self._n_pixels_cached is not None:
            return self._n_pixels_cached

        # Intentar lectura puntual: cachea n_pixels Y wavelengths
        if self.spec is not None:
            try:
                wavelengths_nm, _ = self.spec.read()
                if wavelengths_nm is not None and len(wavelengths_nm) > 0:
                    self._n_pixels_cached = int(len(wavelengths_nm))
                    self._wavelengths_cached_nm = np.asarray(wavelengths_nm, dtype=np.float64)
                    return self._n_pixels_cached
            except Exception:
                pass

        # Fallback: n_pixels sin wavelengths (mock shortcut)
        n = getattr(self.spec, "n_pixels", None)
        if isinstance(n, int) and n > 0:
            return n

        return DEFAULT_N_PIXELS

    def _get_axial_range_m(self) -> Optional[float]:
        """Obtener el rango axial calculado para usarlo en el modo global."""
        if self._axial_range_mm is not None:
            return self._axial_range_mm * 1e-3
        if self._wavelengths_cached_nm is None:
            return None
        try:
            wavelengths_nm = np.asarray(self._wavelengths_cached_nm, dtype=np.float64)
            if wavelengths_nm.ndim != 1 or wavelengths_nm.size < 2:
                return None
            range_mm = ProcessingEngine.theoretical_depth_range_mm(
                float(np.min(wavelengths_nm)),
                float(np.max(wavelengths_nm)),
                int(wavelengths_nm.size),
            )
            self._axial_range_mm = range_mm
            self._update_global_depth_limit_display()
            return range_mm * 1e-3
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _update_global_depth_limit_display(self) -> None:
        """Reflejar el rango axial en la fila de configuración global."""
        if not hasattr(self, "lbl_global_depth_max"):
            return
        if self._axial_range_mm is None:
            self.lbl_global_depth_max.setText("→ -- mm")
            self.lbl_global_depth_max.setToolTip(
                "Rango axial no disponible: se requiere una calibración espectral válida."
            )
            return

        self.lbl_global_depth_max.setText(f"→ {self._axial_range_mm:.2f} mm")
        self.lbl_global_depth_max.setToolTip(
            "Profundidad máxima del modo global.\n"
            "Se toma del rango axial calculado a partir del ancho de banda\n"
            "y del número de muestras espectrales."
        )
        if hasattr(self, "spin_global_depth_min_mm"):
            self.spin_global_depth_min_mm.setRange(0.0, self._axial_range_mm)

    def _cache_wavelengths(self, raw) -> None:
        """Cachear n_pixels y wavelengths desde el primer espectro leído."""
        if self._n_pixels_cached is None and raw.wavelengths_nm is not None:
            self._n_pixels_cached = int(len(raw.wavelengths_nm))
            self._wavelengths_cached_nm = np.asarray(raw.wavelengths_nm, dtype=np.float64)

    def _update_spectrometer_metrics(self, wavelengths_nm=None) -> None:
        """Actualizar resolución axial y rango axial desde la calibración espectral."""
        if wavelengths_nm is None:
            wavelengths_nm = self._wavelengths_cached_nm
        if wavelengths_nm is None:
            self._axial_range_mm = None
            self.lbl_resolution.setText("Resolución axial: -- µm")
            self.lbl_axial_range.setText("Rango axial: -- mm")
            self._update_global_depth_limit_display()
            return

        try:
            wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float64)
            if wavelengths_nm.ndim != 1 or wavelengths_nm.size < 2:
                raise ValueError("Se requieren al menos dos longitudes de onda")
            wavelength_min_nm = float(np.min(wavelengths_nm))
            wavelength_max_nm = float(np.max(wavelengths_nm))
            n_pixels = int(wavelengths_nm.size)
            res_um = ProcessingEngine.theoretical_resolution_um(
                wavelength_min_nm, wavelength_max_nm
            )
            range_mm = ProcessingEngine.theoretical_depth_range_mm(
                wavelength_min_nm, wavelength_max_nm, n_pixels
            )
            self._axial_range_mm = range_mm
            self.lbl_resolution.setText(f"Resolución axial: {res_um:.2f} µm")
            self.lbl_axial_range.setText(f"Rango axial: {range_mm:.2f} mm")
            self._update_global_depth_limit_display()
        except (TypeError, ValueError, ZeroDivisionError):
            self._axial_range_mm = None
            self.lbl_resolution.setText("Resolución axial: -- µm")
            self.lbl_axial_range.setText("Rango axial: -- mm")
            self._update_global_depth_limit_display()

    def _build_scan_config(self) -> ScanConfig:
        """Lee los widgets de barrido y construye un ScanConfig."""
        if not hasattr(self, "scan_checks"):
            return ScanConfig(exposure_ms=DEFAULT_EXPOSURE_MS)

        def sp(axis, idx):
            return self.scan_spinboxes[axis][idx].value()

        return ScanConfig(
            use_x=self.scan_checks[1].isChecked(),
            use_y=self.scan_checks[2].isChecked(),
            use_z=self.scan_checks[3].isChecked(),
            x_start_mm=sp(1, 0),
            x_end_mm=sp(1, 1),
            x_step_mm=sp(1, 2),
            y_start_mm=sp(2, 0),
            y_end_mm=sp(2, 1),
            y_step_mm=sp(2, 2),
            z_start_mm=sp(3, 0),
            z_end_mm=sp(3, 1),
            z_step_mm=sp(3, 2),
            settling_time_s=(
                self.spin_settling_time_s.value()
                if hasattr(self, "spin_settling_time_s")
                else DEFAULT_SETTLING_TIME_S
            ),
            exposure_ms=(
                self.spin_exposure_ms.value()
                if hasattr(self, "spin_exposure_ms")
                else DEFAULT_EXPOSURE_MS
            ),
            scan_mode=(
                self.combo_scan_mode.currentData()
                if hasattr(self, "combo_scan_mode")
                else "snake"
            ),
            axis_order=(
                self.combo_axis_order.currentData()
                if hasattr(self, "combo_axis_order")
                else "XYZ"
            ),
        )

    def _refresh_engine(self):
        """
        Reconstruir el engine cuando cambia la config de la GUI.
        Bloqueado durante SCANNING — la config es inmutable durante un barrido.
        """
        if self._state == SystemState.SCANNING:
            logger.debug("_refresh_engine ignorado: barrido activo")
            return
        try:
            self.engine.update_config(self._build_processing_config())
        except ValueError as exc:
            logger.warning("Configuración de profundidad no disponible: %s", exc)

    # ============================================================
    # UI
    # ============================================================
    def _setup_ui(self):
        self.setWindowTitle(SOFTWARE_NAME)
        self.resize(1280, 800)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ── Panel izquierdo: tabs ─────────────────────────────────
        left_widget = QWidget()
        left_widget.setFixedWidth(420)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(2, 2, 2, 2)
        left_layout.setSpacing(3)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.addTab(self._build_tab_drivers(), "Drivers")
        self._tabs.addTab(self._build_tab_monitor(), "Análisis")
        self._tabs.addTab(self._build_tab_barrido(), "Barrido")
        left_layout.addWidget(self._tabs)

        splitter.addWidget(left_widget)

        # ── Panel derecho: plots ──────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(3)

        self.plot_spec = pg.PlotWidget(title="Espectro")
        self.plot_spec.setLabel("left", "Intensidad")
        self.plot_spec.setLabel("bottom", "λ", units="nm")
        self.plot_spec.showGrid(x=True, y=True, alpha=0.3)

        self.plot_fft = pg.PlotWidget(title="Perfil Axial (CZT)")
        self.plot_fft.setLabel("left", "Amplitud")
        self.plot_fft.setLabel("bottom", "Profundidad", units="mm")
        self.plot_fft.showGrid(x=True, y=True, alpha=0.3)

        # ── Ítems gráficos pre-alocados (rendimiento) ────────────
        # Todos los objetos gráficos se crean UNA vez y se reutilizan
        # en cada frame con setData/setPos/setText/show/hide. Esto
        # elimina el clear() de PlotWidget, que es la operación más
        # cara del ciclo de refresco (~9x más rápido, medido con
        # benchmark offscreen: 9.5 → 1.1 ms/frame).

        # Curvas de perfil: una por ventana posible (MAX_WINDOWS)
        self._profile_curves = []
        for _ in range(MAX_WINDOWS):
            c = pg.PlotDataItem()
            c.hide()
            self.plot_fft.addItem(c)
            self._profile_curves.append(c)

        # Marcadores de picos dominantes: uno por ventana posible,
        # cada uno con el color de su ventana.
        self._peak_markers = []
        for i in range(MAX_WINDOWS):
            m = pg.PlotDataItem(
                pen=None,
                symbol="o",
                symbolSize=10,
                symbolBrush=WINDOW_COLORS[i % len(WINDOW_COLORS)],
            )
            m.hide()
            self.plot_fft.addItem(m)
            self._peak_markers.append(m)

        # Scatter de reflexiones (análisis)
        self._analysis_scatter = pg.PlotDataItem(
            pen=None,
            symbol="x",
            symbolPen=pg.mkPen("w", width=2),
            symbolSize=12,
        )
        self._analysis_scatter.hide()
        self.plot_fft.addItem(self._analysis_scatter)

        # Etiquetas de reflexiones (máximo: 10, el tope del spinbox)
        self._analysis_labels = []
        for _ in range(10):
            lbl = pg.TextItem("", color="w", anchor=(0.5, 1.2))
            lbl.hide()
            self.plot_fft.addItem(lbl)
            self._analysis_labels.append(lbl)

        right_layout.addWidget(self.plot_spec)
        right_layout.addWidget(self.plot_fft)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ── Tabs ─────────────────────────────────────────────────────

    def _compact_grid(self):
        g = QGridLayout()
        g.setContentsMargins(6, 4, 6, 4)
        g.setSpacing(3)
        return g

    @staticmethod
    def _driver_section_label(text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setStyleSheet(
            "QLabel { color: #687384; font-size: 9px; font-weight: 500; "
            "padding: 2px 0 2px; border-bottom: 1px solid #d9dee5; }"
        )
        return label

    @staticmethod
    def _driver_group_style() -> str:
        return (
            "QGroupBox { border: 1px solid #cfd6df; border-radius: 6px; "
            "margin-top: 10px; padding-top: 8px; font-weight: 600; "
            "background-color: #e1e5ea; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; "
            "padding: 0 5px; color: #374151; }"
        )

    @staticmethod
    def _set_status_badge(label: QLabel, text: str, kind: str) -> None:
        colors = {
            "ok": ("#166534", "#dcfce7", "#86efac"),
            "warning": ("#92400e", "#fef3c7", "#fcd34d"),
            "error": ("#991b1b", "#fee2e2", "#fca5a5"),
            "neutral": ("#4b5563", "#f3f4f6", "#d1d5db"),
        }
        foreground, background, border = colors.get(kind, colors["neutral"])
        label.setText(text)
        label.setStyleSheet(
            "QLabel { color: %s; background: %s; border: 1px solid %s; "
            "border-radius: 4px; padding: 3px 7px; font-weight: 600; }"
            % (foreground, background, border)
        )

    def _add_gfa_footer(self, layout):
        """Agregar un pie institucional discreto a una pestaña."""
        pixmap = QPixmap(
            str(Path(__file__).resolve().parent.parent / "assets" / "gfa_logo.png")
        )
        logo = _FooterMark(
            pixmap.scaledToHeight(30, Qt.SmoothTransformation)
            if not pixmap.isNull()
            else QPixmap()
        )
        logo.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo)

    def _build_tab_monitor(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 5, 8, 5)
        v.setSpacing(5)

        # ── Ventanas Profundidad ─────────────────────────────────────────
        g_win = QGroupBox("Ventanas de Profundidad")
        g_win.setStyleSheet(self._driver_group_style())
        lw = self._compact_grid()
        lw.setContentsMargins(2, 3, 2, 3)
        lw.setHorizontalSpacing(2)
        lw.setVerticalSpacing(3)
        g_win.setLayout(lw)
        lw.addWidget(self._driver_section_label("Ventanas CZT"), 0, 0, 1, 4)
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(5)
        header_layout.addWidget(QLabel(""), 0)
        for title in ("Min", "Max", "Paso"):
            header = QLabel(title)
            header.setAlignment(Qt.AlignCenter)
            header_layout.addWidget(header, 1)
        lw.addWidget(header_widget, 1, 0, 1, 4)

        self.window_enabled_checkboxes = []
        self.window_depth_min_spinboxes = []
        self.window_depth_max_spinboxes = []
        self.window_resolution_spinboxes = []
        for i in range(MAX_WINDOWS):
            chk = QCheckBox(f"V{i+1}")
            chk.setToolTip(f"Habilitar ventana {i+1} para análisis por CZT")
            sp_min = QDoubleSpinBox()
            sp_max = QDoubleSpinBox()
            for sp in (sp_min, sp_max):
                sp.setRange(0, 10)
                sp.setDecimals(3)
                sp.setSuffix(" mm")
                sp.setFixedWidth(112)
            sp_min.setValue(i * 0.5)
            sp_max.setValue((i + 1) * 0.5)
            sp_min.setToolTip("Límite inferior de la ventana de profundidad, en mm")
            sp_max.setToolTip("Límite superior de la ventana de profundidad, en mm")
            sp_res = QDoubleSpinBox()
            sp_res.setRange(0.1, 100)
            sp_res.setDecimals(1)
            sp_res.setValue(1.0)
            sp_res.setSingleStep(0.5)
            sp_res.setSuffix(" µm")
            sp_res.setFixedWidth(112)
            sp_res.setToolTip(
                "Paso axial en µm.\n"
                "Determina el espaciado entre puntos en el eje Z\n"
                "dentro de esta ventana."
            )
            chk.stateChanged.connect(self._on_window_changed)
            sp_min.valueChanged.connect(self._on_window_changed)
            sp_max.valueChanged.connect(self._on_window_changed)
            sp_res.valueChanged.connect(self._on_window_changed)
            chk.setFixedWidth(44)
            chk.setStyleSheet("QCheckBox { color: #374151; spacing: 3px; }")
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(5)
            row_layout.addWidget(chk, 0)
            row_layout.addWidget(sp_min, 1)
            row_layout.addWidget(sp_max, 1)
            row_layout.addWidget(sp_res, 1)
            lw.addWidget(row_widget, i + 2, 0, 1, 4)
            self.window_enabled_checkboxes.append(chk)
            self.window_depth_min_spinboxes.append(sp_min)
            self.window_depth_max_spinboxes.append(sp_max)
            self.window_resolution_spinboxes.append(sp_res)
        self.window_enabled_checkboxes[0].setChecked(False)

        # Global: min, max, paso axial
        row_g = MAX_WINDOWS + 2
        lw.addWidget(self._driver_section_label("Modo global"), row_g, 0, 1, 4)
        row_g += 1
        lw.addWidget(QLabel("Global min:"), row_g, 0)
        self.spin_global_depth_min_mm = QDoubleSpinBox()
        self.spin_global_depth_min_mm.setRange(0, 5)
        self.spin_global_depth_min_mm.setDecimals(3)
        self.spin_global_depth_min_mm.setValue(0.05)
        self.spin_global_depth_min_mm.setSingleStep(0.05)
        self.spin_global_depth_min_mm.setSuffix(" mm")
        self.spin_global_depth_min_mm.setFixedWidth(112)
        self.spin_global_depth_min_mm.setToolTip(
            "Profundidad mínima (mm) para el modo global (sin ventanas).\n"
            "Elimina el artefacto DC cerca de Profundidad=0."
        )
        self.spin_global_depth_min_mm.valueChanged.connect(self._on_window_changed)
        lw.addWidget(self.spin_global_depth_min_mm, row_g, 1)
        self.lbl_global_depth_max = QLabel("→ -- mm")
        self.lbl_global_depth_max.setToolTip(
            "Profundidad máxima del modo global, tomada del rango axial espectral."
        )
        lw.addWidget(self.lbl_global_depth_max, row_g, 2)

        self.spin_global_resolution_um = QDoubleSpinBox()
        self.spin_global_resolution_um.setRange(0.1, 100)
        self.spin_global_resolution_um.setDecimals(1)
        self.spin_global_resolution_um.setValue(1.0)
        self.spin_global_resolution_um.setSingleStep(0.5)
        self.spin_global_resolution_um.setSuffix(" µm")
        self.spin_global_resolution_um.setFixedWidth(112)
        self.spin_global_resolution_um.setToolTip(
            "Paso axial global en µm.\n"
            "Determina el espaciado entre puntos en el eje Z\n"
            "cuando no hay ventanas activas."
        )
        self.spin_global_resolution_um.valueChanged.connect(self._on_window_changed)
        lw.addWidget(self.spin_global_resolution_um, row_g, 3)

        v.addWidget(g_win)

        # ── Inspección de picos ────────────────────────────────────
        g_sel = QGroupBox("Inspección de picos")
        g_sel.setStyleSheet(self._driver_group_style())
        ls = self._compact_grid()
        ls.setContentsMargins(10, 5, 10, 5)
        ls.setHorizontalSpacing(6)
        ls.setVerticalSpacing(3)
        g_sel.setLayout(ls)

        # Inspección de reflexiones dominantes (fuera del pipeline).
        # Toggle persistente: mientras está activo, las marcas se
        # recalculan y redibujan en cada refresco del perfil — apto
        # para alineación y monitoreo en vivo.
        ls.addWidget(self._driver_section_label("Reflexiones"), 0, 0, 1, 4)
        self.chk_show_peaks = QCheckBox("Mostrar picos")
        self.chk_show_peaks.setToolTip(
            "Marcar las N reflexiones dominantes del perfil visualizado."
        )
        self.chk_show_peaks.toggled.connect(self._on_analyze_toggled)
        self.spin_n_peaks = QSpinBox()
        self.spin_n_peaks.setRange(1, 10)
        self.spin_n_peaks.setValue(3)
        self.spin_n_peaks.setPrefix("N = ")
        self.spin_n_peaks.setFixedWidth(112)
        self.spin_n_peaks.setToolTip("Cantidad de reflexiones a marcar.")
        self.spin_n_peaks.valueChanged.connect(self._on_analyze_toggled)
        peak_controls = QWidget()
        peak_layout = QHBoxLayout(peak_controls)
        peak_layout.setContentsMargins(0, 0, 0, 0)
        peak_layout.setSpacing(6)
        peak_layout.addWidget(self.chk_show_peaks, 0)
        peak_layout.addStretch(1)
        peak_layout.addWidget(QLabel("Cantidad:"), 0)
        peak_layout.addWidget(self.spin_n_peaks, 0)
        ls.addWidget(peak_controls, 1, 0, 1, 4)

        self.lbl_peak_value = QLabel("Profundidad: --")
        self.lbl_peak_value.setToolTip(
            "Profundidad del pico dominante global (mayor amplitud entre\n"
            "todas las ventanas activas)."
        )
        self._set_status_badge(self.lbl_peak_value, "Profundidad: --", "neutral")
        ls.addWidget(self.lbl_peak_value, 2, 0, 1, 4)
        v.addWidget(g_sel)

        # ── Óptica del cabezal ────────────────────────────────────
        g_opt = QGroupBox("Óptica del cabezal")
        g_opt.setStyleSheet(self._driver_group_style())
        lo = self._compact_grid()
        lo.setContentsMargins(10, 5, 10, 5)
        lo.setHorizontalSpacing(6)
        lo.setVerticalSpacing(3)
        g_opt.setLayout(lo)

        # Inputs
        lo.addWidget(self._driver_section_label("Parámetros"), 0, 0, 1, 4)
        lo.addWidget(QLabel("Ø_f"), 1, 0)
        self.fiber_diameter_spinbox = QDoubleSpinBox()
        self.fiber_diameter_spinbox.setRange(1, 50)
        self.fiber_diameter_spinbox.setDecimals(1)
        self.fiber_diameter_spinbox.setValue(DEFAULT_FIBER_DIAMETER_UM)
        self.fiber_diameter_spinbox.setSingleStep(0.5)
        self.fiber_diameter_spinbox.setSuffix(" µm")
        self.fiber_diameter_spinbox.setToolTip(
            "Diámetro del núcleo de la fibra óptica en µm.\n"
            "Para monomodo SM600: ~5 µm.\n"
            "Idealmente usar MFD (mode field diameter)."
        )
        self.fiber_diameter_spinbox.valueChanged.connect(self._update_optics)
        lo.addWidget(self.fiber_diameter_spinbox, 1, 1)

        lo.addWidget(QLabel("λ"), 1, 2)
        self.wavelength_central_spinbox = QDoubleSpinBox()
        self.wavelength_central_spinbox.setRange(400, 2000)
        self.wavelength_central_spinbox.setDecimals(0)
        self.wavelength_central_spinbox.setValue(DEFAULT_WAVELENGTH_NM)
        self.wavelength_central_spinbox.setSingleStep(10)
        self.wavelength_central_spinbox.setSuffix(" nm")
        self.wavelength_central_spinbox.setToolTip(
            "Longitud de onda central de la fuente en nm.\n"
            "Típico para OCT: 850 nm (banda ancha)."
        )
        self.wavelength_central_spinbox.valueChanged.connect(self._update_optics)
        lo.addWidget(self.wavelength_central_spinbox, 1, 3)

        lo.addWidget(QLabel("f_col"), 2, 0)
        self.collimator_focal_length_spinbox = QDoubleSpinBox()
        self.collimator_focal_length_spinbox.setRange(0.5, 200)
        self.collimator_focal_length_spinbox.setDecimals(1)
        self.collimator_focal_length_spinbox.setValue(DEFAULT_COLLIMATOR_FOCAL_LENGTH_MM)
        self.collimator_focal_length_spinbox.setSingleStep(1)
        self.collimator_focal_length_spinbox.setSuffix(" mm")
        self.collimator_focal_length_spinbox.setToolTip(
            "Distancia focal del colimador en mm.\n"
            "Determina el diámetro del haz colimado."
        )
        self.collimator_focal_length_spinbox.valueChanged.connect(self._update_optics)
        lo.addWidget(self.collimator_focal_length_spinbox, 2, 1)

        lo.addWidget(QLabel("f_obj"), 2, 2)
        self.objective_focal_length_spinbox = QDoubleSpinBox()
        self.objective_focal_length_spinbox.setRange(0.5, 200)
        self.objective_focal_length_spinbox.setDecimals(1)
        self.objective_focal_length_spinbox.setValue(DEFAULT_OBJECTIVE_FOCAL_LENGTH_MM)
        self.objective_focal_length_spinbox.setSingleStep(1)
        self.objective_focal_length_spinbox.setSuffix(" mm")
        self.objective_focal_length_spinbox.setToolTip(
            "Distancia focal del objetivo en mm.\n"
            "Determina la magnificación y el spot en la muestra."
        )
        self.objective_focal_length_spinbox.valueChanged.connect(self._update_optics)
        lo.addWidget(self.objective_focal_length_spinbox, 2, 3)

        # Unificar el ancho de los controles numéricos de esta pestaña.
        # El ancho permite conservar completos los valores y sus sufijos.
        for spinbox in (
            self.fiber_diameter_spinbox,
            self.wavelength_central_spinbox,
            self.collimator_focal_length_spinbox,
            self.objective_focal_length_spinbox,
        ):
            spinbox.setFixedWidth(112)

        # Outputs (2 columnas) — sin estilos custom, misma fuente que el resto
        self.lbl_opt_beam = QLabel("Ø_col: --")
        self.lbl_opt_beam.setToolTip("Diámetro del haz después del colimador (mm)")
        self.lbl_opt_na = QLabel("NA: --")
        self.lbl_opt_na.setToolTip(
            "Apertura numérica efectiva en la muestra.\n" "NA = D_beam / (2 · f_obj)"
        )
        self.lbl_opt_geo = QLabel("Ø_focal: --")
        self.lbl_opt_geo.setToolTip(
            "Diámetro de cintura gaussiana en el plano focal (µm).\n"
            "Se calcula propagando W0 hasta Wz y resolviendo la cintura Ws.\n"
            "Ø_focal = 2 · Ws"
        )
        self.lbl_opt_confocal = QLabel("b_conf: --")
        self.lbl_opt_confocal.setToolTip(
            "Parámetro confocal en µm.\n"
            "Rango efectivo alrededor del foco."
        )

        lo.addWidget(self._driver_section_label("Resultados"), 3, 0, 1, 4)
        lo.addWidget(self.lbl_opt_beam, 4, 0, 1, 2)
        lo.addWidget(self.lbl_opt_geo, 4, 2, 1, 2)
        lo.addWidget(self.lbl_opt_na, 5, 0, 1, 2)
        lo.addWidget(self.lbl_opt_confocal, 5, 2, 1, 2)

        v.addWidget(g_opt)

        # Calcular valores iniciales
        self._update_optics()
        self._update_spectrometer_metrics()

        v.addStretch()
        self._add_gfa_footer(v)
        return w

    def _build_tab_drivers(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 5, 8, 5)
        v.setSpacing(5)

        # ── Espectrómetro + visualización ──────────────────────────
        g_spec = QGroupBox("Espectrómetro")
        g_spec.setStyleSheet(self._driver_group_style())
        ls = self._compact_grid()
        ls.setContentsMargins(10, 5, 10, 5)
        ls.setHorizontalSpacing(6)
        ls.setVerticalSpacing(4)
        for column in range(4):
            ls.setColumnStretch(column, 1)
        g_spec.setLayout(ls)
        la = ls
        la.addWidget(self._driver_section_label("Visualización"), 4, 0, 1, 4)

        la.addWidget(QLabel("Exposición"), 5, 0)
        self.spin_exposure_ms = QDoubleSpinBox()
        self.spin_exposure_ms.setRange(0.1, 2000)
        self.spin_exposure_ms.setValue(DEFAULT_EXPOSURE_MS)
        self.spin_exposure_ms.setDecimals(1)
        self.spin_exposure_ms.setSuffix(" ms")
        self.spin_exposure_ms.valueChanged.connect(self._on_exposure_changed)
        self.spin_exposure_ms.setToolTip(
            "Tiempo de integración del espectrómetro en milisegundos."
        )
        la.addWidget(self.spin_exposure_ms, 5, 1, 1, 3)

        self.btn_start_monitor = QPushButton("▶ Iniciar")
        self.btn_start_monitor.setToolTip(
            "Iniciar modo monitor: visualización continua.\n"
            "Muestra espectro y perfil axial en tiempo real."
        )
        self.btn_start_monitor.clicked.connect(self.start_monitor)
        self.btn_stop_monitor = QPushButton("■ Detener")
        self.btn_stop_monitor.setToolTip("Detener modo monitor")
        self.btn_stop_monitor.clicked.connect(self.stop_monitor)
        self.btn_stop_monitor.setEnabled(False)
        monitor_button_row = QHBoxLayout()
        monitor_button_row.setSpacing(6)
        monitor_button_row.addWidget(self.btn_start_monitor, 1)
        monitor_button_row.addWidget(self.btn_stop_monitor, 1)
        la.addLayout(monitor_button_row, 6, 0, 1, 4)

        self.chk_dark = QCheckBox("Dark")
        self.chk_dark.setToolTip(
            "Activar corrección de dark.\n"
            "Resta el promedio de los electric dark pixels del CCD\n"
            "para eliminar el offset por corriente oscura."
        )
        self.chk_dark.toggled.connect(self._on_dark_toggled)
        la.addWidget(self.chk_dark, 7, 0)
        self.chk_nonlinearity = QCheckBox("No-linealidad")
        self.chk_nonlinearity.setToolTip(
            "Activar corrección de no-linealidad del detector.\n"
            "Aplica los coeficientes polinomiales almacenados\n"
            "en la EEPROM del espectrómetro (calibración de fábrica).\n"
            "Corrige la respuesta no-lineal del CCD."
        )
        self.chk_nonlinearity.toggled.connect(self._on_nonlinearity_toggled)
        la.addWidget(self.chk_nonlinearity, 7, 1)

        self.lbl_resolution = QLabel("Resolución axial: -- µm")
        self.lbl_resolution.setToolTip(
            "Resolución axial teórica (FWHM) en µm.\n"
            "Calculada a partir del ancho de banda espectral."
        )
        la.addWidget(self.lbl_resolution, 8, 0, 1, 4)

        self.lbl_axial_range = QLabel("Rango axial: -- mm")
        self.lbl_axial_range.setToolTip(
            "Rango axial máximo no ambiguo del espectrómetro.\n"
            "Representa la profundidad máxima del vector medible.\n"
            "Se calcula con z_max = π/(2·Δk), donde Δk depende\n"
            "del ancho de banda y del número de píxeles."
        )
        la.addWidget(self.lbl_axial_range, 9, 0, 1, 4)

        self.lbl_system_state = QLabel("Estado: IDLE")
        self._set_status_badge(self.lbl_system_state, "Estado: IDLE", "neutral")
        self.lbl_system_state.setToolTip(
            "Estado actual del sistema:\n"
            "IDLE = listo, esperando acción.\n"
            "MONITOR = visualización en tiempo real.\n"
            "SCANNING = barrido en curso.\n"
            "ERROR = error detectado, requiere atención."
        )
        la.addWidget(self.lbl_system_state, 10, 0, 1, 4)


        g_motor = QGroupBox("Motores")
        g_motor.setStyleSheet(self._driver_group_style())
        lm = self._compact_grid()
        lm.setContentsMargins(10, 5, 10, 5)
        lm.setHorizontalSpacing(6)
        lm.setVerticalSpacing(4)
        for column in range(4):
            lm.setColumnStretch(column, 1)
        g_motor.setLayout(lm)

        # ── Conexión ───────────────────────────────────────────────
        lm.addWidget(self._driver_section_label("Conexión"), 0, 0, 1, 4)
        controller_label = QLabel("Driver:")
        controller_label.setMinimumWidth(92)
        lm.addWidget(controller_label, 1, 0)
        self.combo_motion_controller = QComboBox()
        for kind, label in list_motion_controllers():
            self.combo_motion_controller.addItem(label, kind)
        self.combo_motion_controller.setToolTip(
            "Controlador de motores registrado.\n"
            "Al cambiarlo, se crea el driver seleccionado mediante la factory."
        )
        self._set_combo_data(self.combo_motion_controller, self._motion_controller_kind)
        self.combo_motion_controller.currentIndexChanged.connect(
            self._on_motion_controller_selected
        )
        lm.addWidget(self.combo_motion_controller, 1, 1, 1, 3)

        lm.addWidget(QLabel("Puerto:"), 2, 0)
        self.combo_port = QComboBox()
        for p in ["COM3", "COM4", "COM5", "COM6", "COM7", "COM8"]:
            self.combo_port.addItem(p)
        self.combo_port.setEditable(True)
        self.combo_port.setCurrentText(DEFAULT_MOTOR_PORT)
        self.combo_port.setToolTip(
            "Puerto serie del controlador de motores.\n"
            "Seleccionar o escribir el puerto correcto (ej: COM3)."
        )
        lm.addWidget(self.combo_port, 2, 1, 1, 2)
        self.btn_motor_reconnect = QPushButton("Reconectar")
        self.btn_motor_reconnect.setMinimumWidth(98)
        self.btn_motor_reconnect.setToolTip(
            "Reconectar motores en el puerto y controlador seleccionados"
        )
        self.btn_motor_reconnect.clicked.connect(self._reconnect_motors)
        lm.addWidget(self.btn_motor_reconnect, 2, 3)
        self.lbl_motor_status = QLabel("Estado: --")
        self._set_status_badge(self.lbl_motor_status, "Estado: --", "neutral")
        self.lbl_motor_status.setToolTip(
            "Estado de conexión del controlador de motores"
        )
        lm.addWidget(self.lbl_motor_status, 3, 0, 1, 4)
        lm.addWidget(QLabel("Tolerancia:"), 4, 0)
        self.spin_position_tolerance_um = QDoubleSpinBox()
        self.spin_position_tolerance_um.setDecimals(3)
        self.spin_position_tolerance_um.setRange(
            MIN_POSITION_TOLERANCE_UM,
            1_000_000.0,
        )
        self.spin_position_tolerance_um.setSingleStep(0.1)
        self.spin_position_tolerance_um.setValue(DEFAULT_POSITION_TOLERANCE_UM)
        self.spin_position_tolerance_um.setSuffix(" µm")
        self.spin_position_tolerance_um.setToolTip(
            "Error máximo permitido entre la posición objetivo y la posición real.\n"
            "Se aplica a movimientos manuales y barridos.\n"
            "Por defecto: 2 µm. Mínimo configurable: 0,5 µm."
        )
        lm.addWidget(self.spin_position_tolerance_um, 4, 1, 1, 3)

        lm.addWidget(self._driver_section_label("Estado real"), 5, 0, 1, 4)
        btn_home = QPushButton("Home", clicked=self._move_home)
        btn_home.setToolTip(
            "Mover todos los ejes a la posición 0.0 mm:\n"
            "X = 0.0 mm, Y = 0.0 mm, Z = 0.0 mm"
        )
        lm.addWidget(btn_home, 6, 0)
        self._manual_move_buttons.append(btn_home)
        self.motor_toggle_buttons = {}
        self.motor_status_labels = {}
        for col, (name, axis) in enumerate([("X", 1), ("Y", 2), ("Z", 3)]):
            btn = QPushButton(f"Habilitar {name}")
            btn.setToolTip(
                f"Habilitar el eje {name}: activa la corriente del motor.\n"
                "Necesario antes de mover."
            )
            btn.clicked.connect(lambda _, a=axis: self._toggle_motor(a))
            lm.addWidget(btn, 6, col + 1)
            self.motor_toggle_buttons[axis] = btn

        self.btn_motion_stop = QPushButton("STOP")
        self.btn_motion_stop.setStyleSheet(
            "QPushButton { background-color: #b00020; color: white; "
            "font-weight: bold; }"
        )
        self.btn_motion_stop.setToolTip("Detener el movimiento manual actual.")
        self.btn_motion_stop.clicked.connect(self._stop_manual_move)
        self.btn_motion_stop.setEnabled(False)
        lm.addWidget(self.btn_motion_stop, 7, 0)
        for col, (name, axis) in enumerate([("X", 1), ("Y", 2), ("Z", 3)]):
            status = QLabel("● OFF")
            status.setAlignment(Qt.AlignCenter)
            status.setStyleSheet("color: #777777; font-weight: bold;")
            status.setToolTip(f"Estado del motor {name}")
            lm.addWidget(status, 7, col + 1)
            self.motor_status_labels[axis] = status

        self._update_motor_axis_controls()
        self._set_manual_buttons_enabled(True)

        # ── Posición real dentro de Estado real ────────────────────
        self.lbl_pos_actual = QLabel("Posición real:")
        self.lbl_pos_actual.setStyleSheet("font-size: 10pt;")
        self.lbl_pos_actual.setToolTip(
            "Posición real leída desde los motores; se actualiza periódicamente."
        )
        lm.addWidget(self.lbl_pos_actual, 8, 0)
        self.motor_position_labels = {}
        for col, (name, axis) in enumerate([("X", 1), ("Y", 2), ("Z", 3)]):
            position_label = QLabel(f"{name}=-- mm")
            position_label.setAlignment(Qt.AlignCenter)
            position_label.setStyleSheet("font-size: 8pt;")
            position_label.setToolTip(f"Posición real del eje {name} en mm")
            lm.addWidget(position_label, 8, col + 1)
            self.motor_position_labels[axis] = position_label

        # ── Movimiento manual ──────────────────────────────────────
        lm.addWidget(self._driver_section_label("Movimiento manual"), 9, 0, 1, 4)
        self.manual_pos = {}
        for row, (name, axis) in enumerate([("X", 1), ("Y", 2), ("Z", 3)]):
            visual_row = row + 10
            lm.addWidget(QLabel(f"{name}"), visual_row, 0)
            sp = QDoubleSpinBox()
            sp.setRange(-100, 100)
            sp.setDecimals(4)
            sp.setSingleStep(0.001)
            sp.setSuffix(" mm")
            sp.setMinimumWidth(140)
            sp.setToolTip(f"Posición objetivo del eje {name} en mm")
            self.manual_pos[axis] = sp
            btn = QPushButton(f"Mover {name}")
            btn.setMinimumWidth(100)
            btn.setMaximumWidth(136)
            btn.setToolTip(f"Mover eje {name} a la posición indicada y esperar")
            btn.clicked.connect(lambda _, a=axis: self._move_axis_manual(a))
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            row_layout.addWidget(sp, 1)
            row_layout.addWidget(btn, 0)
            lm.addWidget(row_widget, visual_row, 1, 1, 3)
            self._manual_move_buttons.append(btn)
        # ── Conexión del espectrómetro ──────────────────────────────
        ls.addWidget(self._driver_section_label("Conexión"), 0, 0, 1, 4)
        ls.addWidget(QLabel("Detector:"), 1, 0)
        self.combo_spectrometer = QComboBox()
        for kind, label in list_spectrometers():
            self.combo_spectrometer.addItem(label, kind)
        self.combo_spectrometer.setToolTip(
            "Espectrómetro registrado.\n"
            "Al cambiarlo, se crea el driver seleccionado mediante la factory."
        )
        self._set_combo_data(self.combo_spectrometer, self._spectrometer_kind)
        self.combo_spectrometer.currentIndexChanged.connect(
            self._on_spectrometer_selected
        )
        ls.addWidget(self.combo_spectrometer, 1, 1, 1, 3)

        self.btn_spec_reconnect = QPushButton("Buscar / conectar")
        self.btn_spec_reconnect.setToolTip(
            "Reconectar el espectrómetro tras una desconexión USB."
        )
        self.btn_spec_reconnect.clicked.connect(self._reconnect_spectrometer)
        ls.addWidget(self.btn_spec_reconnect, 2, 1, 1, 3)

        self.lbl_spec_status = QLabel("Estado: --")
        self._set_status_badge(self.lbl_spec_status, "Estado: --", "neutral")
        self.lbl_spec_status.setToolTip(
            "Estado de conexión del espectrómetro."
        )
        ls.addWidget(self.lbl_spec_status, 3, 0, 1, 4)
        v.addWidget(g_spec)
        v.addWidget(g_motor)

        # Estado inicial del hardware
        self._update_motor_status()
        self._update_spec_status()
        self._update_spectrometer_metrics()
        self._apply_hardware_capabilities()

        # En modo mock: deshabilitar controles de reconexión
        # (_set_state corre antes de que estos widgets existan,
        # así que el bloqueo inicial se aplica acá).
        if self._use_mock:
            self.btn_motor_reconnect.setEnabled(False)
            self.combo_motion_controller.setEnabled(False)
            self.combo_port.setEnabled(False)
            self.btn_spec_reconnect.setEnabled(False)
            self.combo_spectrometer.setEnabled(False)

        v.addStretch()
        self._add_gfa_footer(v)
        return w

    def _build_tab_barrido(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 5, 8, 5)
        v.setSpacing(5)

        # ── Nombre de muestra ─────────────────────────────────────
        g_sample = QGroupBox("Muestra")
        g_sample.setStyleSheet(self._driver_group_style())
        ls_sample = self._compact_grid()
        ls_sample.setContentsMargins(10, 5, 10, 5)
        ls_sample.setHorizontalSpacing(6)
        ls_sample.setVerticalSpacing(3)
        g_sample.setLayout(ls_sample)
        ls_sample.addWidget(self._driver_section_label("Identificación"), 0, 0, 1, 4)
        ls_sample.addWidget(QLabel("Nombre"), 1, 0)
        self.edit_sample_name = QLineEdit("Muestra")
        self.edit_sample_name.setToolTip(
            "Nombre de la muestra. Se usa como prefijo del archivo guardado."
        )
        ls_sample.addWidget(self.edit_sample_name, 1, 1, 1, 3)
        v.addWidget(g_sample)

        # ── Adquisición por punto ─────────────────────────────────
        g_acq = QGroupBox("Adquisición por punto")
        g_acq.setStyleSheet(self._driver_group_style())
        la = self._compact_grid()
        la.setContentsMargins(10, 5, 10, 5)
        la.setHorizontalSpacing(6)
        la.setVerticalSpacing(3)
        g_acq.setLayout(la)

        la.addWidget(self._driver_section_label("Adquisición"), 0, 0, 1, 4)
        la.addWidget(QLabel("M (mediciones):"), 1, 0)
        self.spin_measurements_per_point = QSpinBox()
        self.spin_measurements_per_point.setRange(1, 1000)
        self.spin_measurements_per_point.setValue(1)
        self.spin_measurements_per_point.setToolTip(
            "Cantidad de mediciones válidas por punto.\n"
            "Cada medición es procesada de forma independiente\n"
            "(sin promedios). Se conservan los M espectros,\n"
            "M perfiles y M sets de picos por punto.\n"
            "La primera lectura SIEMPRE se descarta (doc §4)."
        )
        self.spin_measurements_per_point.valueChanged.connect(self._on_m_changed)
        la.addWidget(self.spin_measurements_per_point, 1, 1, 1, 3)
        v.addWidget(g_acq)

        # ── Barrido ───────────────────────────────────────────────
        g = QGroupBox("Barrido")
        g.setStyleSheet(self._driver_group_style())
        ls = self._compact_grid()
        ls.setContentsMargins(2, 3, 2, 3)
        ls.setHorizontalSpacing(2)
        ls.setVerticalSpacing(3)
        g.setLayout(ls)
        self.scan_spinboxes = {}
        self.scan_checks = {}
        ls.addWidget(self._driver_section_label("Ejes"), 0, 0, 1, 4)
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(2)
        header_layout.addWidget(QLabel(""), 0)
        for title in ("Inicio", "Fin", "Paso"):
            header = QLabel(title)
            header.setAlignment(Qt.AlignCenter)
            header_layout.addWidget(header, 1)
        ls.addWidget(header_widget, 1, 0, 1, 4)

        for row, (name, axis, de) in enumerate(
            [("X", 1, 1.0), ("Y", 2, 0.0), ("Z", 3, 0.0)], start=2
        ):
            chk = QCheckBox(name)
            chk.setToolTip(f"Activar eje {name} para el barrido")
            sp_in = QDoubleSpinBox()
            sp_in.setRange(-100, 100)
            sp_in.setDecimals(4)
            sp_in.setSingleStep(0.001)
            sp_in.setSuffix(" mm")
            sp_in.setMinimumWidth(119)
            sp_in.setToolTip(f"Posición inicial del eje {name} (mm)")
            sp_end = QDoubleSpinBox()
            sp_end.setRange(-100, 100)
            sp_end.setDecimals(4)
            sp_end.setValue(de)
            sp_end.setSingleStep(0.001)
            sp_end.setSuffix(" mm")
            sp_end.setMinimumWidth(119)
            sp_end.setToolTip(f"Posición final del eje {name} (mm)")
            sp_step = QDoubleSpinBox()
            sp_step.setRange(0.0001, 10)
            sp_step.setValue(0.1)
            sp_step.setDecimals(4)
            sp_step.setSingleStep(0.001)
            sp_step.setSuffix(" mm")
            sp_step.setMinimumWidth(119)
            sp_step.setToolTip(f"Paso entre puntos del eje {name} (mm)")
            chk.setFixedWidth(30)
            chk.setStyleSheet("QCheckBox { color: #374151; spacing: 3px; }")
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(2)
            row_layout.addWidget(chk, 0)
            row_layout.addWidget(sp_in, 1)
            row_layout.addWidget(sp_end, 1)
            row_layout.addWidget(sp_step, 1)
            ls.addWidget(row_widget, row, 0, 1, 4)
            self.scan_spinboxes[axis] = (sp_in, sp_end, sp_step)
            self.scan_checks[axis] = chk

        self._update_scan_axis_checks()

        ls.addWidget(self._driver_section_label("Recorrido"), 5, 0, 1, 4)
        self.spin_settling_time_s = QDoubleSpinBox()
        self.spin_settling_time_s.setSuffix(" s")
        self.spin_settling_time_s.setRange(0, 9999.999)
        self.spin_settling_time_s.setValue(0.050)
        self.spin_settling_time_s.setSingleStep(0.010)
        self.spin_settling_time_s.setDecimals(3)
        self.spin_settling_time_s.setToolTip(
            "Tiempo de espera después de cada movimiento (s).\n"
            "Permite que las vibraciones mecánicas se disipen\n"
            "antes de adquirir el espectro.\n"
            "Rango permitido: 0 a 9999.999 s (2 h 46 min 40 s)."
        )
        self.combo_scan_mode = QComboBox()
        self.combo_scan_mode.addItem("Snake", "snake")
        self.combo_scan_mode.addItem("Raster", "raster")
        self.combo_scan_mode.setToolTip(
            "Patrón de recorrido del barrido:\n"
            "Snake: ida y vuelta (boustrophedon, más eficiente).\n"
            "Raster: siempre en la misma dirección."
        )
        self.combo_axis_order = QComboBox()
        self.combo_axis_order.addItem("X → Y → Z", "XYZ")
        self.combo_axis_order.addItem("X → Z → Y", "XZY")
        self.combo_axis_order.addItem("Y → X → Z", "YXZ")
        self.combo_axis_order.addItem("Y → Z → X", "YZX")
        self.combo_axis_order.addItem("Z → X → Y", "ZXY")
        self.combo_axis_order.addItem("Z → Y → X", "ZYX")
        self.combo_axis_order.setToolTip(
            "Orden del barrido: rápido → medio → lento.\n"
            "El primer eje es el que barre (inner loop).\n"
            "Ej: 'X → Y → Z' = X barre, Y avanza, Z es lento."
        )
        route_row = QWidget()
        route_layout = QHBoxLayout(route_row)
        route_layout.setContentsMargins(0, 0, 0, 0)
        route_layout.setSpacing(6)
        route_layout.addWidget(QLabel("Espera"), 0)
        route_layout.addWidget(self.spin_settling_time_s, 0)
        route_layout.addSpacing(10)
        route_layout.addWidget(QLabel("Modo:"), 0)
        route_layout.addWidget(self.combo_scan_mode, 1)
        ls.addWidget(route_row, 6, 0, 1, 4)

        order_row = QWidget()
        order_layout = QHBoxLayout(order_row)
        order_layout.setContentsMargins(0, 0, 0, 0)
        order_layout.setSpacing(6)
        order_layout.addWidget(QLabel("Orden:"), 0)
        order_layout.addWidget(self.combo_axis_order, 1)
        ls.addWidget(order_row, 7, 0, 1, 4)

        self.chk_pause_plots = QCheckBox("Pausar gráficos")
        self.chk_pause_plots.setToolTip(
            "Desactiva la actualización de gráficos durante el barrido.\n"
            "Puede mejorar la velocidad en barridos grandes."
        )
        ls.addWidget(self.chk_pause_plots, 8, 0, 1, 2)
        v.addWidget(g)

        # ── Guardar ───────────────────────────────────────────────
        g_save = QGroupBox("Guardar")
        g_save.setStyleSheet(self._driver_group_style())
        ls_s = self._compact_grid()
        ls_s.setContentsMargins(10, 5, 10, 5)
        ls_s.setHorizontalSpacing(6)
        ls_s.setVerticalSpacing(3)
        g_save.setLayout(ls_s)
        ls_s.addWidget(self._driver_section_label("Guardado"), 0, 0, 1, 4)

        # Espectros: checkbox simple. M=1 → siempre se guarda el espectro
        # único como (1, N_pixels) para mantener consistencia de shape.
        self.chk_save_spectra = QCheckBox("Espectros")
        self.chk_save_spectra.setChecked(True)
        self.chk_save_spectra.setToolTip(
            "Guardar el espectro crudo del punto en el archivo de salida.\n"
            "Shape: (n_points, 1, n_pixels) — 1 espectro por punto."
        )
        # Picos
        self.chk_save_peaks = QCheckBox("Picos (Profundidad)")
        self.chk_save_peaks.setChecked(True)
        self.chk_save_peaks.setToolTip(
            "Guardar posiciones de picos detectados (Profundidad en metros).\n"
            "Incluye posición y amplitud por ventana."
        )
        save_checks = QWidget()
        save_checks_layout = QHBoxLayout(save_checks)
        save_checks_layout.setContentsMargins(0, 0, 0, 0)
        save_checks_layout.setSpacing(12)
        save_checks_layout.addWidget(self.chk_save_spectra, 1)
        save_checks_layout.addWidget(self.chk_save_peaks, 1)
        ls_s.addWidget(save_checks, 1, 0, 1, 4)

        # Perfil axial: checkbox + combo (Módulo / Complejo)
        self.chk_save_profile = QCheckBox("Perfil axial")
        self.chk_save_profile.setChecked(False)
        self.chk_save_profile.setToolTip(
            "Guardar el perfil axial (resultado de la CZT)"
        )
        self.chk_save_profile.stateChanged.connect(self._on_save_profile_toggled)
        self.combo_profile_mode = QComboBox()
        self.combo_profile_mode.addItem("Módulo |z|", "modulus")
        self.combo_profile_mode.addItem("Complejo (Re+Im)", "complex")
        self.combo_profile_mode.setCurrentIndex(0)
        self.combo_profile_mode.setMinimumWidth(150)
        self.combo_profile_mode.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo_profile_mode.setEnabled(False)
        self.combo_profile_mode.setToolTip(
            "Módulo: amplitud del perfil (mitad de espacio).\n"
            "Complejo: Re+Im, permite reprocesamiento de fase offline."
        )
        profile_row = QWidget()
        profile_layout = QHBoxLayout(profile_row)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        profile_layout.setSpacing(8)
        profile_layout.addWidget(self.chk_save_profile, 0)
        profile_layout.addWidget(self.combo_profile_mode, 1)
        ls_s.addWidget(profile_row, 2, 0, 1, 4)

        # Formato
        self.save_format_combo_box = QComboBox()
        self.save_format_combo_box.addItem("NPZ (.npz)", "npz")
        self.save_format_combo_box.addItem("HDF5 (.h5)", "h5")
        self.save_format_combo_box.setToolTip(
            "NPZ: archivo numpy, sin dependencias externas.\n"
            "HDF5: escritura incremental, recomendado para\n"
            "barridos grandes (requiere h5py)."
        )
        format_row = QWidget()
        format_layout = QHBoxLayout(format_row)
        format_layout.setContentsMargins(0, 0, 0, 0)
        format_layout.setSpacing(8)
        format_layout.addWidget(QLabel("Formato:"), 0)
        format_layout.addWidget(self.save_format_combo_box, 1)
        ls_s.addWidget(format_row, 3, 0, 1, 4)
        v.addWidget(g_save)

        # ── Botones + progreso ────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.btn_run_scan = QPushButton("▶ Ejecutar")
        self.btn_run_scan.setToolTip("Iniciar el barrido con la configuración actual")
        self.btn_run_scan.clicked.connect(self.run_scan)
        self.btn_abort_scan = QPushButton("⏹ ABORTAR")
        self.btn_abort_scan.setToolTip("Detener el barrido en curso")
        self.btn_abort_scan.clicked.connect(self.abort_scan)
        self.btn_abort_scan.setEnabled(False)
        self.btn_abort_scan.setStyleSheet(
            "QPushButton{background-color:#b42318;color:white;font-weight:bold;"
            "border: 1px solid #8f1d14;border-radius: 4px;padding: 4px 8px;}"
        )
        btn_row.addWidget(self.btn_run_scan, 1)
        btn_row.addWidget(self.btn_abort_scan, 1)
        v.addLayout(btn_row)

        progress_row = QWidget()
        progress_layout = QHBoxLayout(progress_row)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(8)
        progress_layout.addWidget(QLabel("Progreso"), 0)
        self.scan_progress_bar = QProgressBar()
        self.scan_progress_bar.setRange(0, 100)
        self.scan_progress_bar.setValue(0)
        progress_layout.addWidget(self.scan_progress_bar, 1)
        v.addWidget(progress_row)
        self.lbl_scan_progress = QLabel("Listo")
        self.lbl_scan_progress.setWordWrap(True)
        self.lbl_scan_progress.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._set_status_badge(self.lbl_scan_progress, "Listo", "neutral")
        v.addWidget(self.lbl_scan_progress)
        v.addStretch()
        self._add_gfa_footer(v)
        return w

    # ── Callbacks de widgets ──────────────────────────────────────

    def _on_window_changed(self, _val=None):
        self._refresh_engine()

    def _on_m_changed(self, _val=None):
        self._refresh_engine()

    def _sync_profile_controls(self):
        """Sincronizar el combo de perfil axial con el checkbox y el estado."""
        if not hasattr(self, "combo_profile_mode") or not hasattr(
            self, "chk_save_profile"
        ):
            return
        self.combo_profile_mode.setEnabled(
            self.chk_save_profile.isChecked()
            and self._state != SystemState.SCANNING
        )

    def _on_save_profile_toggled(self, _state):
        self._sync_profile_controls()

    def _update_optics(self, _val=None):
        """Recalcular parámetros ópticos cuando cambia un input."""
        try:
            result = calculate_optics(
                fiber_diameter_um=self.fiber_diameter_spinbox.value(),
                wavelength_nm=self.wavelength_central_spinbox.value(),
                collimator_focal_length_mm=self.collimator_focal_length_spinbox.value(),
                objective_focal_length_mm=self.objective_focal_length_spinbox.value(),
            )
            self.lbl_opt_beam.setText(f"Ø_col: {result.beam_diameter_mm:.2f} mm")
            self.lbl_opt_na.setText(f"NA: {result.na_effective:.2f}")
            self.lbl_opt_geo.setText(f"Ø_focal: {result.spot_gaussian_um:.4f} µm")
            self.lbl_opt_confocal.setText(
                f"b_conf: {result.confocal_parameter_um:.2f} µm"
            )
        except (ValueError, ZeroDivisionError):
            for lbl in (
                self.lbl_opt_beam,
                self.lbl_opt_na,
                self.lbl_opt_geo,
                self.lbl_opt_confocal,
            ):
                lbl.setText(lbl.text().split(":")[0] + ": --")

    # ============================================================
    # MODO MONITOR — thread separado
    # ============================================================
    def _apply_live_correction(self, checkbox, setter, enabled, label):
        """Aplicar una corrección del espectrómetro sin reiniciar adquisición."""
        if self.spec is None:
            return
        try:
            setter(bool(enabled))
        except Exception as e:
            logger.error("No se pudo cambiar %s: %s", label, e)
            checkbox.blockSignals(True)
            checkbox.setChecked(not enabled)
            checkbox.blockSignals(False)
            self._update_spec_status()
            QMessageBox.warning(
                self,
                "Error de espectrómetro",
                f"No se pudo cambiar {label}: {e}",
            )

    def _on_exposure_changed(self, value: float):
        """Aplicar el tiempo de integración sin reiniciar adquisición."""
        if self.spec is None or not self._spectrometer_capabilities()["exposure_control"]:
            return

        value = float(value)
        previous_exposure_ms = self._applied_exposure_ms
        try:
            self.spec.set_exposure_ms(value)
            self._applied_exposure_ms = value
        except Exception as e:
            logger.error("No se pudo cambiar el tiempo de exposición: %s", e)
            self.spec = None
            self.acq.set_spectrometer(None)
            self._update_spec_status()
            self.spin_exposure_ms.blockSignals(True)
            self.spin_exposure_ms.setValue(previous_exposure_ms)
            self.spin_exposure_ms.blockSignals(False)
            QMessageBox.warning(
                self,
                "Error de espectrómetro",
                f"No se pudo cambiar el tiempo de exposición: {e}",
            )

    def _on_dark_toggled(self, enabled: bool):
        if (
            self.spec is None
            or not self._spectrometer_capabilities()["dark_correction"]
        ):
            return
        self._apply_live_correction(
            self.chk_dark,
            self.spec.set_dark_correction,
            enabled,
            "la corrección Dark",
        )

    def _on_nonlinearity_toggled(self, enabled: bool):
        if (
            self.spec is None
            or not self._spectrometer_capabilities()["nonlinearity_correction"]
        ):
            return
        self._apply_live_correction(
            self.chk_nonlinearity,
            self.spec.set_nonlinearity_correction,
            enabled,
            "la corrección de linealidad",
        )

    def start_monitor(self):
        if self._state not in (SystemState.IDLE, SystemState.ERROR):
            return
        if self.spec is None:
            QMessageBox.warning(self, "Error", "Espectrómetro no conectado")
            return

        caps = self._spectrometer_capabilities()
        if not caps["wavelengths_nm"]:
            QMessageBox.warning(
                self,
                "Operación no disponible",
                "El espectrómetro no declara wavelengths_nm.",
            )
            return

        # Configurar hardware — puede detectar desconexión USB
        try:
            self._apply_spectrometer_settings()
        except Exception as e:
            self.spec = None
            self.acq.set_spectrometer(None)
            self._update_spec_status()
            QMessageBox.warning(self, "Error", f"Espectrómetro desconectado: {e}")
            return

        # Actualizar engines
        self._refresh_engine()
        self.acq.update_config(self._build_scan_config())

        # Crear y arrancar worker
        self._monitor_worker = MonitorWorker(self.acq, self.engine)
        self._monitor_worker.result_ready.connect(self._on_monitor_result)
        self._monitor_worker.error_occurred.connect(self._on_monitor_error)
        self._monitor_worker.finished.connect(self._on_monitor_thread_finished)
        self._monitor_worker.start()

        self._set_state(SystemState.MONITORING)
        logger.info("Modo Monitor iniciado (thread)")

    def stop_monitor(self):
        if self._monitor_worker is not None:
            self._monitor_worker.stop()
            if not self._monitor_worker.wait(5000):
                logger.error("Monitor no terminó dentro del timeout; se conserva la referencia al thread")
                return
            self._monitor_worker = None
        self._set_state(SystemState.IDLE)
        logger.info("Modo Monitor detenido")

    def _on_monitor_result(self, raw, result):
        """Slot: recibe resultado del MonitorWorker. Solo grafica."""
        wavelengths_nm = raw.wavelengths_nm
        # FIX: M=1 forzado → no se promedia. Se grafica el espectro crudo.
        intensities = raw.spectra[0]

        self._cache_wavelengths(raw)
        self._update_spectrometer_metrics(wavelengths_nm)

        self.plot_spec.plot(wavelengths_nm, intensities, clear=True, pen=SPECTRUM_COLOR)
        self._plot_result(result)

    def _on_monitor_error(self, msg):
        logger.error("Monitor error: %s", msg)

    def _on_monitor_thread_finished(self):
        """
        Slot: QThread del MonitorWorker terminó.

        Dos escenarios posibles:
        1. stop_monitor() ya limpió todo → _monitor_worker es None → no-op.
        2. Auto-stop por errores consecutivos → _monitor_worker todavía
           existe → transicionar a ERROR y actualizar estado de hardware.

        La distinción funciona porque stop_monitor() es síncrono: llama
        wait(), nullifica _monitor_worker y cambia estado ANTES de que
        el event loop entregue esta señal.
        """
        if self._monitor_worker is None:
            return
        # Auto-stop: el thread terminó sin que nadie llamó stop_monitor().
        self._monitor_worker = None
        self._update_spec_status()
        self._set_state(SystemState.ERROR)

    # ============================================================
    # Graficación de ProcessingResult
    # ============================================================
    def _plot_result(self, result: ProcessingResult):
        """
        Grafica la medición i=0 del ProcessingResult.

        El pipeline procesa y conserva M mediciones independientes,
        pero para visualización en vivo la GUI usa solo la primera.
        Todas las mediciones se guardan en disco vía el saver.

        Sin clear(): todos los ítems gráficos son persistentes y se
        actualizan con setData/setPos/show/hide para evitar el costo
        de destruir/recrear la escena Qt en cada frame (~9x más rápido).
        """
        self._last_result = result

        # Ocultar todo por default; se muestran solo los que tienen datos.
        for c in self._profile_curves:
            c.hide()
        for m in self._peak_markers:
            m.hide()
        self._analysis_scatter.hide()
        for lbl in self._analysis_labels:
            lbl.hide()

        if result.measurements_per_point == 0:
            self.lbl_peak_value.setText("Profundidad: --")
            return

        # Usar siempre la primera medición para graficar.
        wr_dict = result.window_results[0]
        pk_dict = result.peaks[0]

        # Curvas de perfil: una por ventana activa, todas con el
        # mismo peso visual (diferenciadas por color de ventana).
        for win_id, wr in wr_dict.items():
            if len(wr.depth_axis_m) == 0 or win_id >= len(self._profile_curves):
                continue
            depth_mm = wr.depth_axis_m * 1e3
            amp = np.abs(wr.profile)
            col = (
                GLOBAL_PROFILE_COLOR
                if not result.is_windowed
                else WINDOW_COLORS[win_id % len(WINDOW_COLORS)]
            )
            curve = self._profile_curves[win_id]
            curve.setData(depth_mm, amp)
            curve.setPen(pg.mkPen(col, width=2))
            curve.show()

        # Pico dominante de CADA ventana activa (círculo en su color).
        # El label muestra el Profundidad del pico global dominante (mayor
        # amplitud entre todas las ventanas).
        best_pk = None
        for win_id, pk in pk_dict.items():
            if pk is None or pk.depth_m is None:
                continue
            if win_id < len(self._peak_markers):
                peak_col = (
                    GLOBAL_PROFILE_COLOR
                    if not result.is_windowed
                    else WINDOW_COLORS[win_id % len(WINDOW_COLORS)]
                )
                self._peak_markers[win_id].setSymbolBrush(pg.mkBrush(peak_col))
                self._peak_markers[win_id].setData(
                    [pk.depth_m * 1e3],
                    [pk.amplitude],
                )
                self._peak_markers[win_id].show()
            if best_pk is None or pk.amplitude > best_pk.amplitude:
                best_pk = pk

        if best_pk is not None:
            self.lbl_peak_value.setText(f"Profundidad: {best_pk.depth_m*1e6:.2f} µm")
        else:
            self.lbl_peak_value.setText("Profundidad: --")

        # Overlay de reflexiones dominantes (toggle activo): busca
        # las N reflexiones más importantes en TODAS las ventanas
        # activas.
        if self.chk_show_peaks.isChecked():
            self._draw_peak_analysis(wr_dict)

    def _draw_peak_analysis(self, wr_dict):
        """
        Marcar las N reflexiones dominantes de TODAS las ventanas activas.
        """
        n_requested = self.spin_n_peaks.value()

        all_pos = []
        all_amp = []
        for _wid, wr in wr_dict.items():
            if wr is None or len(wr.depth_axis_m) == 0:
                continue
            pos, amp = analyze_profile_peaks(
                wr.depth_axis_m,
                np.abs(wr.profile),
                n_peaks=n_requested,
            )
            if len(pos) > 0:
                all_pos.append(pos)
                all_amp.append(amp)

        if not all_pos:
            return

        all_pos = np.concatenate(all_pos)
        all_amp = np.concatenate(all_amp)

        order = np.argsort(all_amp)[::-1][:n_requested]
        positions = all_pos[order]
        amplitudes = all_amp[order]

        self._analysis_scatter.setData(positions * 1e3, amplitudes)
        self._analysis_scatter.show()

        for i, (pos, amp) in enumerate(zip(positions, amplitudes)):
            lbl = self._analysis_labels[i]
            lbl.setText(f"{i+1}: {pos*1e3:.3f} mm")
            lbl.setPos(pos * 1e3, amp)
            lbl.show()

    def _on_analyze_toggled(self, *_):
        """
        Redibujar al activar/desactivar el análisis o cambiar N.

        Con el monitor corriendo el próximo frame ya refleja el
        cambio; este redraw inmediato cubre el caso de un perfil
        estático (monitor detenido).
        """
        if self._last_result is not None:
            self._plot_result(self._last_result)

    # ============================================================
    # Motores manuales
    # ============================================================
    def _set_manual_buttons_enabled(self, enabled: bool):
        """Habilita/deshabilita los botones según estado y capacidades."""
        can_move = (
            self.mot is not None
            and self._motor_capabilities()["absolute_move"]
            and self._motor_capabilities()["position_readback"]
        )
        for btn in self._manual_move_buttons:
            btn.setEnabled(enabled and can_move)
        if hasattr(self, "motor_toggle_buttons"):
            can_toggle = (
                enabled
                and self.mot is not None
                and self._motor_capabilities()["axis_enable"]
            )
            for axis, button in self.motor_toggle_buttons.items():
                button.setEnabled(can_toggle and axis in self.available_axes)
        if hasattr(self, "btn_motion_stop"):
            self.btn_motion_stop.setEnabled(
                not enabled
                and self._motor_capabilities()["stop_motion"]
                and self._move_worker is not None
            )

    def _start_move_worker(self, moves):
        """
        Lanza un MoveWorker con la lista de movimientos (axis, target).
        Bloquea nuevos movimientos hasta que termine (worker activo).
        """
        if self.mot is None:
            QMessageBox.warning(self, "Error", "Motores no conectados")
            return
        motor_caps = self._motor_capabilities()
        if not motor_caps["absolute_move"] or not motor_caps["position_readback"]:
            QMessageBox.warning(
                self,
                "Operación no disponible",
                "El controlador no declara movimiento absoluto y lectura de posición.",
            )
            return
        if self._move_worker is not None and self._move_worker.isRunning():
            # Ya hay un movimiento en curso: ignorar para evitar carreras.
            logger.info("Movimiento manual en curso — clic ignorado")
            return

        worker = MoveWorker(
            self.mot,
            moves,
            position_tolerance_mm=self.spin_position_tolerance_um.value() * 1e-3,
        )
        worker.finished_ok.connect(self._on_manual_move_finished)
        worker.stopped.connect(self._on_manual_move_stopped)
        worker.error_occurred.connect(self._on_manual_move_error)
        # Limpieza automática del thread cuando termine.
        worker.finished.connect(self._on_manual_move_thread_finished)

        # Guardar referencia para evitar garbage collection.
        self._move_worker = worker

        # UI: deshabilitar botones y mostrar estado.
        self._set_manual_buttons_enabled(False)
        self.lbl_pos_actual.setText("Posición real: MOVING…")

        worker.start()
        self._set_state(self._state)

    def _stop_manual_move(self):
        """Detener el movimiento manual en curso sin deshabilitar ejes."""
        if self._move_worker is None or not self._move_worker.isRunning():
            return
        self.lbl_pos_actual.setText("Posición real: DETENIENDO…")
        self._move_worker.stop()

    def _move_axis_manual(self, axis: int):
        if self.mot is None:
            QMessageBox.warning(self, "Error", "Motores no conectados")
            return
        if axis not in self.available_axes:
            name = AXIS_NAMES.get(axis, "?")
            QMessageBox.warning(self, "Error", f"Eje {name} no disponible")
            return
        pos = self.manual_pos[axis].value()
        self._start_move_worker([(axis, pos)])

    def _move_home(self):
        if self.mot is None:
            QMessageBox.warning(self, "Error", "Motores no conectados")
            return
        moves = [(axis, 0.0) for axis in sorted(self.available_axes)]
        if not moves:
            QMessageBox.warning(self, "Error", "No hay ejes disponibles")
            return
        self._start_move_worker(moves)

    def _on_manual_move_finished(self, results):
        """
        Slot: el MoveWorker terminó OK.
        results: lista [(axis, actual_pos), ...].
        """
        self.lbl_pos_actual.setText("Posición real:")
        for axis, actual in results:
            name = AXIS_NAMES.get(axis, "?")
            position_label = getattr(self, "motor_position_labels", {}).get(axis)
            if position_label is not None:
                position_label.setText(f"{name}={actual:.4f}")

    def _on_manual_move_error(self, msg: str):
        """Slot: el MoveWorker reportó error."""
        self.lbl_pos_actual.setText(f"Posición real: ERROR: {msg}")
        QMessageBox.critical(self, "Error de Motor", msg)

    def _on_manual_move_stopped(self):
        """Slot: el movimiento manual fue detenido por el usuario."""
        self.lbl_pos_actual.setText("Posición real: DETENIDO")

    def _on_manual_move_thread_finished(self):
        """Slot: el QThread del MoveWorker terminó (siempre se llama)."""
        self._set_manual_buttons_enabled(True)
        self._set_state(self._state)
        # No nullificamos _move_worker inmediatamente: dejamos que viva
        # hasta el próximo movimiento o el cierre, para evitar destrucción
        # mientras Qt aún procesa señales.
        QTimer.singleShot(0, self._sync_motor_positions)

    def _toggle_motor(self, axis):
        if self._motor_enabled.get(axis, False):
            self._disable_motor(axis)
        else:
            self._enable_motor(axis)

    def _enable_motor(self, axis):
        if self.mot is None:
            QMessageBox.warning(self, "Error", "Motores no conectados")
            return
        if axis not in self.available_axes:
            QMessageBox.warning(self, "Error", "Eje no disponible")
            return
        try:
            self.mot.enable_axis(axis)
            self._motor_enabled[axis] = True
            self._update_motor_axis_controls()
            self._update_scan_axis_checks()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _disable_motor(self, axis):
        if self.mot is None:
            QMessageBox.warning(self, "Error", "Motores no conectados")
            return
        if axis not in self.available_axes:
            QMessageBox.warning(self, "Error", "Eje no disponible")
            return
        try:
            self.mot.disable_axis(axis)
            self._motor_enabled[axis] = False
            self._update_motor_axis_controls()
            self._update_scan_axis_checks()
            logger.info("Motor %s deshabilitado", AXIS_NAMES[axis])
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _update_motor_axis_controls(self):
        """Actualizar el toggle y el indicador visual de cada motor."""
        if not hasattr(self, "motor_toggle_buttons"):
            return

        for axis, button in self.motor_toggle_buttons.items():
            name = AXIS_NAMES[axis]
            enabled = self._motor_enabled.get(axis, False)
            button.setEnabled(
                self.mot is not None
                and self._motor_capabilities()["axis_enable"]
                and axis in self.available_axes
                and self._state != SystemState.SCANNING
                and not (
                    self._move_worker is not None
                    and self._move_worker.isRunning()
                )
            )
            if enabled:
                button.setText(f"Deshabilitar {name}")
                button.setToolTip(
                    f"Deshabilitar el eje {name}: quita la corriente del motor.\n"
                    "El eje queda libre (se puede mover a mano)."
                )
                status_text = "● ON"
                status_style = "color: #198754; font-weight: bold;"
            else:
                button.setText(f"Habilitar {name}")
                button.setToolTip(
                    f"Habilitar el eje {name}: activa la corriente del motor.\n"
                    "Necesario antes de mover."
                )
                status_text = "● OFF"
                status_style = "color: #777777; font-weight: bold;"

            status = self.motor_status_labels.get(axis)
            if status is not None:
                status.setText(status_text)
                status.setStyleSheet(status_style)

    def _update_scan_axis_checks(self):
        """Sincronizar las tildes de barrido con el estado de cada motor."""
        if not hasattr(self, "scan_checks"):
            return

        available_axes = self.available_axes if self.mot is not None else set()
        motor_caps = self._motor_capabilities()
        is_scanning = getattr(self, "_state", None) == SystemState.SCANNING

        for axis, checkbox in self.scan_checks.items():
            motor_ready = (
                axis in available_axes
                and self._motor_enabled.get(axis, False)
                and motor_caps["absolute_move"]
                and motor_caps["position_readback"]
            )
            if not motor_ready and checkbox.isChecked():
                checkbox.setChecked(False)
            checkbox.setEnabled(motor_ready and not is_scanning)

    def _sync_motor_positions(self):
        """Actualizar la lectura real sin sobrescribir las consignas manuales."""
        if (
            self.mot is None
            or not self._motor_capabilities()["position_readback"]
            or getattr(self, "_state", None) == SystemState.SCANNING
            or (
                self._move_worker is not None
                and self._move_worker.isRunning()
            )
        ):
            return
        for axis in sorted(self.available_axes):
            try:
                pos = self.mot.get_position(axis)
                name = AXIS_NAMES[axis]
                position_label = getattr(self, "motor_position_labels", {}).get(axis)
                if position_label is not None:
                    position_label.setText(f"{name}={pos:.4f} mm")
            except Exception as e:
                logger.warning("Eje %d: error leyendo posición: %s", axis, e)
        if self.available_axes:
            self.lbl_pos_actual.setText("Posición real:")

    def _update_motor_status(self):
        """Actualizar el label de estado de motores."""
        if self.mot is None:
            self._set_status_badge(self.lbl_motor_status, "No conectado", "error")
            self.lbl_motor_status.setToolTip("Controlador de motores no conectado.")
            return
        if self._use_mock:
            controller_name = type(self.mot).__name__
            self._set_status_badge(
                self.lbl_motor_status,
                f"Conectado · {controller_name}",
                "ok",
            )
            self.lbl_motor_status.setToolTip(
                f"Controlador de motores conectado: {controller_name}."
            )
            return
        port = (
            self.combo_port.currentText().strip()
            if hasattr(self, "combo_port")
            else "?"
        )
        controller_name = type(self.mot).__name__
        self._set_status_badge(
            self.lbl_motor_status,
            f"Conectado · {controller_name}",
            "ok",
        )
        self.lbl_motor_status.setToolTip(
            f"Controlador conectado en {port}: {controller_name}."
        )

    def _reconnect_motors(self):
        if self._move_worker is not None and self._move_worker.isRunning():
            logger.warning("Reconexión de motores ignorada: movimiento manual en curso")
            return
        port = self.combo_port.currentText().strip()
        if not port:
            QMessageBox.warning(self, "Error", "Ingrese un puerto")
            return
        if self.mot is not None:
            try:
                self.mot.close()
            except Exception:
                pass
        self._motor_enabled = {1: False, 2: False, 3: False}
        self._update_motor_axis_controls()
        self._update_scan_axis_checks()
        self._set_status_badge(
            self.lbl_motor_status,
            f"Conectando {self._motion_controller_kind} · {port}...",
            "warning",
        )
        self.mot = self._init_motors(
            port=port,
            kind=self._motion_controller_kind,
        )
        if self.mot is not None:
            controller_name = type(self.mot).__name__
            self._set_status_badge(
                self.lbl_motor_status,
                f"Conectado · {controller_name}",
                "ok",
            )
            self.lbl_motor_status.setToolTip(
                f"Controlador conectado en {port}: {controller_name}."
            )
            self._sync_motor_positions()
        else:
            self._set_status_badge(
                self.lbl_motor_status,
                f"Fallo en {port}.",
                "error",
            )
        self._apply_hardware_capabilities()

    def _reconnect_spectrometer(self):
        """Crear y conectar el espectrómetro seleccionado."""
        self._set_status_badge(self.lbl_spec_status, "Buscando...", "warning")
        self.lbl_spec_status.repaint()

        if self.spec is not None:
            try:
                self.spec.close()
            except Exception:
                pass
        self.spec = None

        ok = False
        try:
            spec = create_spectrometer(self._spectrometer_kind)
            if spec.connect():
                self.spec = spec
                ok = True
        except Exception as e:
            logger.warning(
                "Reconexión de espectrómetro %s fallida: %s",
                self._spectrometer_kind,
                e,
            )

        if ok:
            self.acq.set_spectrometer(self.spec)
            self._n_pixels_cached = None
            self._wavelengths_cached_nm = None
            self._update_spectrometer_metrics(None)
            self._apply_spectrometer_settings()
        else:
            self.acq.set_spectrometer(None)
        self._apply_hardware_capabilities()
        self._update_spec_status()

    def _update_spec_status(self):
        """Actualizar el label de estado del espectrómetro (display-only)."""
        if self.spec is None:
            self._set_status_badge(self.lbl_spec_status, "No conectado", "error")
            self.lbl_spec_status.setToolTip("Espectrómetro no conectado.")
            return
        if self._use_mock:
            info = self.spec.device_info()
            model = info.get("model", type(self.spec).__name__)
            self._set_status_badge(
                self.lbl_spec_status,
                f"Conectado · {model}",
                "ok",
            )
            self.lbl_spec_status.setToolTip(
                f"Espectrómetro conectado: {model} · serial "
                f"{info.get('serial', '?')}."
            )
            return
        info = self.spec.device_info()
        if info:
            model = info.get("model", "?")
            serial = info.get("serial", "?")
            self._set_status_badge(
                self.lbl_spec_status,
                f"Conectado · {model}",
                "ok",
            )
            self.lbl_spec_status.setToolTip(
                f"Espectrómetro conectado: {model} · serial {serial}."
            )
        else:
            # device_info vacío puede ser transitorio (ej: USB inestable
            # justo después de reconectar). No destruir self.spec — el
            # próximo read() o set_exposure_ms() confirmará si funciona.
            self._set_status_badge(
                self.lbl_spec_status,
                "Conectado · sin información",
                "warning",
            )

    # ============================================================
    # BARRIDO — thread separado
    # ============================================================
    def run_scan(self):
        if self._state == SystemState.MONITORING:
            logger.info("Deteniendo monitor antes de iniciar el barrido")
            self.stop_monitor()
            if self._state != SystemState.IDLE:
                QMessageBox.warning(
                    self,
                    "Monitor ocupado",
                    "No se pudo detener la visualización antes de iniciar el barrido.",
                )
                return

        if self._state != SystemState.IDLE:
            QMessageBox.warning(
                self, "Error", f"No se puede iniciar en estado {self._state.name}."
            )
            return
        if self.mot is None:
            QMessageBox.warning(self, "Error", "Motores no conectados")
            return
        if self.spec is None:
            QMessageBox.warning(self, "Error", "Espectrómetro no conectado")
            return

        scan_cfg = self._build_scan_config()
        use_x, use_y, use_z = scan_cfg.use_x, scan_cfg.use_y, scan_cfg.use_z

        if not any([use_x, use_y, use_z]):
            QMessageBox.warning(self, "Error", "Seleccione al menos un eje")
            return

        for axis, name, use in [(1, "X", use_x), (2, "Y", use_y), (3, "Z", use_z)]:
            if use and axis not in self.available_axes:
                QMessageBox.critical(
                    self, "Error", f"Eje {name} seleccionado pero no conectado."
                )
                return

        # Validaciones de UI
        errors = []
        for axis, name, use in [(1, "X", use_x), (2, "Y", use_y), (3, "Z", use_z)]:
            if use:
                _, _, sp_step = self.scan_spinboxes[axis]
                if sp_step.value() <= 0:
                    errors.append(f"Eje {name}: step debe ser > 0")
        for i in range(MAX_WINDOWS):
            if self.window_enabled_checkboxes[i].isChecked():
                if self.window_depth_min_spinboxes[i].value() >= self.window_depth_max_spinboxes[i].value():
                    errors.append(f"Ventana V{i+1}: Min debe ser < Max")
        if errors:
            QMessageBox.critical(self, "Config inválida", "\n".join(errors))
            return

        caps = self._spectrometer_capabilities()
        if not caps["wavelengths_nm"]:
            QMessageBox.warning(
                self,
                "Operación no disponible",
                "El espectrómetro no declara wavelengths_nm.",
            )
            return

        # Aplicar sólo las configuraciones soportadas por el driver.
        try:
            self._apply_spectrometer_settings()
        except Exception as e:
            self.spec = None
            self.acq.set_spectrometer(None)
            self._update_spec_status()
            QMessageBox.warning(self, "Error", f"Espectrómetro desconectado: {e}")
            return

        save_cfg = SaveConfig(
            save_spectra=self.chk_save_spectra.isChecked(),
            save_peaks=self.chk_save_peaks.isChecked(),
            save_profile=self.chk_save_profile.isChecked(),
            profile_mode=self.combo_profile_mode.currentData(),
        )
        if not any([save_cfg.save_spectra, save_cfg.save_peaks, save_cfg.save_profile]):
            QMessageBox.warning(
                self, "Guardado", "Seleccioná al menos una opción de guardado."
            )
            return

        # Construir config y sincronizar engines
        try:
            proc_cfg = self._build_processing_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Configuración inválida", str(exc))
            return
        self.engine.update_config(proc_cfg)
        self.acq.update_config(scan_cfg)

        # Garantizar wavelengths_nm antes de armar metadata (Schema 6.0).
        if self._wavelengths_cached_nm is None:
            try:
                wavelengths_nm, _ = self.spec.read()
                self._wavelengths_cached_nm = np.asarray(wavelengths_nm, dtype=np.float64)
                self._n_pixels_cached = int(len(wavelengths_nm))
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"No se pudo obtener calibración espectral:\n{e}\n\n"
                    "No se puede iniciar un scan sin wavelengths.",
                )
                return

        sample_name = self.edit_sample_name.text()
        file_format = self.save_format_combo_box.currentData()
        filepath = generate_filename(file_format=file_format, sample_name=sample_name)

        # Congelar valores de GUI que la metadata necesita.
        # Se leen UNA vez acá y se pasan como dict cerrado.
        # _prepare_metadata ya NO lee widgets directamente.
        self._frozen_gui_params = {
            "sample_name": sample_name,
            "exposure_ms": self.spin_exposure_ms.value(),
            "position_tolerance_um": self.spin_position_tolerance_um.value(),
            "position_tolerance_mm": self.spin_position_tolerance_um.value() * 1e-3,
            "dark_enabled": self.chk_dark.isChecked(),
            "nonlinearity_enabled": self.chk_nonlinearity.isChecked(),
            "optics_inputs": {
                "fiber_diameter_um": self.fiber_diameter_spinbox.value(),
                "wavelength_nm": self.wavelength_central_spinbox.value(),
                "collimator_focal_length_mm": self.collimator_focal_length_spinbox.value(),
                "objective_focal_length_mm": self.objective_focal_length_spinbox.value(),
            },
        }

        # Metadata inicial.
        metadata = self._prepare_metadata(scan_cfg, proc_cfg)

        self.scan_progress_bar.setValue(0)
        axes = scan_cfg.active_axes
        if axes:
            # active_axes es rápido→lento.
            order_str = " → ".join(axes)
            fast = axes[0]
            self.lbl_scan_progress.setText(f"Barrido: {order_str} (rápido: {fast})")
        else:
            self.lbl_scan_progress.setText("Barrido en curso...")
        self._set_state(SystemState.SCANNING)

        ok = self._scan_controller.start(
            mot=self.mot,
            scan_cfg=scan_cfg,
            proc_cfg=proc_cfg,
            save_cfg=save_cfg,
            metadata=metadata,
            filepath=filepath,
            file_format=file_format,
            n_pixels=self._get_n_pixels(),
            position_tolerance_mm=self._frozen_gui_params["position_tolerance_mm"],
            wavelengths_nm=self._wavelengths_cached_nm,
        )
        if not ok:
            # El controller emitió scan_error; _on_scan_error() ya reaccionó.
            return

    def _on_scan_point(self, x_mm, y_mm, z_mechanical_mm, raw, result):
        """
        Slot: el controller notifica un punto ya guardado.
        La GUI solo grafica (no maneja buffers ni saver).
        """
        self._cache_wavelengths(raw)

        if not self.chk_pause_plots.isChecked():
            try:
                # FIX: M=1 forzado → graficar espectro crudo, no promedio.
                self.plot_spec.plot(
                    raw.wavelengths_nm, raw.spectra[0], clear=True, pen=SPECTRUM_COLOR
                )
                self._plot_result(result)
            except Exception as e:
                logger.error(
                    "Error graficando punto (%s,%s,%s) mm: %s",
                    x_mm,
                    y_mm,
                    z_mechanical_mm,
                    e,
                    exc_info=True,
                )

    def _on_scan_progress(
        self,
        acquired: int,
        total: int,
        eta_str: str,
        x_mm: float,
        y_mm: float,
        z_mechanical_mm: float,
    ):
        """Slot: actualizar barra de progreso y label. Recibe todo por señal."""
        pct = int(100 * acquired / max(total, 1)) if total else 0
        self.scan_progress_bar.setValue(pct)

        # Mostrar posición en el orden de ejes activos (rápido al final)
        pos_str = (
            f"X={x_mm:.3f} Y={y_mm:.3f} "
            f"Z={z_mechanical_mm:.4f} mm"
        )

        self.lbl_scan_progress.setText(
            f"Punto {acquired}/{total} — {pos_str} — {eta_str}"
        )

    def _resume_idle(self):
        self.scan_progress_bar.setValue(0)
        self._set_state(SystemState.IDLE)

    def _on_scan_finished(self, filepath: str, summary):
        # Mantener visible el resultado final. El reset a 0% ocurre al
        # iniciar el siguiente scan, no inmediatamente después de terminar.
        self.scan_progress_bar.setValue(100)
        self._set_state(SystemState.IDLE)
        self._show_scan_result(filepath or None, summary)

    def _on_scan_error(self, error_msg: str):
        self._set_state(SystemState.ERROR)
        self.scan_progress_bar.setValue(0)
        self.lbl_scan_progress.setText(f"ERROR: {error_msg}")
        QMessageBox.critical(
            self, "Error de Motor", f"Barrido abortado:\n\n{error_msg}"
        )

    def abort_scan(self):
        if self._scan_controller.is_running():
            self._scan_controller.abort()
            self.lbl_scan_progress.setText("Abortando...")

    def _on_scan_aborted(self, filepath: str, summary):
        self._resume_idle()
        if summary.acquired_points > 0:
            self._show_scan_result(filepath or None, summary)
        else:
            self.lbl_scan_progress.setText("ABORTADO: sin datos")

    # ============================================================
    # Guardado
    # ============================================================
    def _prepare_metadata(
        self,
        scan_cfg: ScanConfig,
        proc_cfg: ProcessingConfig,
        start_time: Optional[datetime] = None,
    ) -> dict:
        """
        Construye metadata del scan (Schema 6.0).

        Lee de self._frozen_gui_params (congelados en run_scan),
        NO de widgets directamente. Esto garantiza que los valores
        sean inmutables durante todo el scan.
        """
        gui = self._frozen_gui_params

        # Ventanas: Schema 6.0 siempre tiene ≥1 entrada.
        # Modo global (sin ventanas activas) = ventana única.
        windows = [
            {"depth_min_m": w.depth_min_m, "depth_max_m": w.depth_max_m} for w in proc_cfg.active_windows
        ]
        if not windows:
            windows = [
                {
                    "depth_min_m": proc_cfg.depth_min_global_m,
                    "depth_max_m": proc_cfg.depth_max_global_m,
                }
            ]

        # Hardware: modelo y serial del espectrómetro
        spec_info = self.spec.device_info() if self.spec is not None else {}

        effective_start = start_time or datetime.now()

        return {
            # Identificación
            "software_version": SOFTWARE_VERSION,
            "sample_name": gui["sample_name"],
            # Hardware
            "spectrometer_model": spec_info.get("model", ""),
            "spectrometer_serial": spec_info.get("serial", ""),
            # Adquisición
            "exposure_ms": gui["exposure_ms"],
            "position_tolerance_um": gui["position_tolerance_um"],
            "position_tolerance_mm": gui["position_tolerance_mm"],
            "dark_enabled": gui["dark_enabled"],
            "nonlinearity_enabled": gui["nonlinearity_enabled"],
            # Barrido
            "scan_mode": scan_cfg.scan_mode,
            "axis_order": scan_cfg.axis_order,
            # Ventanas (siempre ≥1)
            "windows": windows,
            # Estado
            "planned_points": scan_cfg.planned_points,
            "start_time": effective_start,
            # Óptica (solo inputs)
            "optics": gui["optics_inputs"],
        }

    def _show_scan_result(self, filepath: Optional[str], summary):
        """Diálogo de fin de scan. summary es un ScanSummary del controller."""
        status = "ABORTADO" if summary.aborted else "✓ Completado"
        fname = os.path.basename(filepath or "desconocido")
        msg = (
            f"{status}\n\n"
            f"Puntos: {summary.acquired_points}/{summary.planned_points} "
            f"({summary.completion_pct:.1f}%)\n"
            f"Archivo: {filepath}"
        )
        if summary.aborted:
            QMessageBox.warning(self, "Barrido Abortado", msg)
        else:
            QMessageBox.information(self, "Éxito", msg)
        self.lbl_scan_progress.setText(f"{status}: {fname}")

    # ============================================================
    # Cierre
    # ============================================================
    def closeEvent(self, event):
        # Detener workers
        if self._monitor_worker is not None:
            self._monitor_worker.stop()
            if not self._monitor_worker.wait(2000):
                logger.error("No se puede cerrar: MonitorWorker sigue activo")
                event.ignore()
                return
        if self._move_worker is not None and self._move_worker.isRunning():
            # No hay forma de abortar goto_and_wait limpiamente; esperamos.
            logger.info("Esperando fin de movimiento manual antes de cerrar")
            if not self._move_worker.wait(5000):
                logger.error("No se puede cerrar: MoveWorker sigue activo")
                event.ignore()
                return
        if self._scan_controller.is_running():
            self._scan_controller.abort()
            if not self._scan_controller.wait_finish(5000):
                logger.error("No se puede cerrar: ScanWorker sigue activo")
                event.ignore()
                return

        if self.saver.is_open():
            logger.warning("Cerrando archivo abierto al salir")
            self.saver.close_scan(aborted=True)
        if self.mot:
            try:
                self.mot.close()
            except Exception:
                pass
        if self.spec:
            try:
                self.spec.close()
            except Exception:
                pass
        event.accept()


class _FooterMark(QLabel):
    def __init__(self, pixmap):
        super().__init__()
        self.setPixmap(pixmap)
        self.setCursor(Qt.ArrowCursor)
        self.setToolTip("")
        self._clicks = 0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._reset)

    def mousePressEvent(self, event):
        self._clicks += 1
        self._timer.start(2000)
        if self._clicks >= 10:
            self._reset()
            self._show_message()
        super().mousePressEvent(event)

    def _show_message(self):
        self._popup = QLabel(self.window())
        self._popup.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        self._popup.setAttribute(Qt.WA_ShowWithoutActivating)
        self._popup.setText(
            "Si algo falla, consulten a Asuka.\n"
            "Si Asuka no sabe, Luca sabe menos.\n\n"
            "PD: reinicien y vuelvan a probar, cruzando los dedos.\n"
            "Suerte."
        )
        self._popup.setStyleSheet(
            "QLabel { background: #20242b; color: #f0f0f0; "
            "border: 1px solid #596273; border-radius: 5px; "
            "padding: 8px; }"
        )
        self._popup.adjustSize()
        pos = self.mapToGlobal(self.rect().topLeft())
        pos.setY(max(0, pos.y() - self._popup.height() - 8))
        self._popup.move(pos)
        self._popup.show()
        QTimer.singleShot(5000, self._popup.close)

    def _reset(self):
        self._clicks = 0
