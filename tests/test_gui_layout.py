from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QGroupBox, QLabel

from gui.main_gui import (
    GLOBAL_PROFILE_COLOR,
    SPECTRUM_COLOR,
    WINDOW_COLORS,
    OCTGUI,
)
from gui.workers import MonitorWorker, SystemState
from processing.engine import ProcessingEngine


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_gui_tabs_and_acquisition_group_order(qapp):
    gui = OCTGUI(use_mock=True)
    gui.show()
    qapp.processEvents()
    try:
        assert gui.windowTitle() == "OCT_Static_Software"
        assert [gui._tabs.tabText(i) for i in range(gui._tabs.count())] == [
            "Drivers",
            "Análisis",
            "Barrido",
        ]

        drivers_groups = [
            group.title()
            for group in gui._tabs.widget(0).findChildren(QGroupBox)
        ]
        monitor_groups = [
            group.title()
            for group in gui._tabs.widget(1).findChildren(QGroupBox)
        ]
        assert drivers_groups == [
            "Espectrómetro",
            "Motores",
        ]
        assert monitor_groups == [
            "Ventanas de Profundidad",
            "Inspección de picos",
            "Óptica del cabezal",
        ]
        section_groups = (
            gui._tabs.widget(0).findChildren(QGroupBox)
            + gui._tabs.widget(1).findChildren(QGroupBox)
            + gui._tabs.widget(2).findChildren(QGroupBox)
        )
        assert len(section_groups) == 9
        assert all("background-color: #e1e5ea" in group.styleSheet() for group in section_groups)
        monitor_section_labels = {
            label.text()
            for label in gui._tabs.widget(1).findChildren(QLabel)
            if label.text() in {
                "VENTANAS CZT",
                "MODO GLOBAL",
                "REFLEXIONES",
                "PARÁMETROS",
                "RESULTADOS",
            }
        }
        assert monitor_section_labels == {
            "VENTANAS CZT",
            "MODO GLOBAL",
            "REFLEXIONES",
            "PARÁMETROS",
            "RESULTADOS",
        }
        assert hasattr(gui, "spin_n_peaks")
        assert gui.spin_n_peaks.value() == 3
        assert gui.spin_n_peaks.prefix() == "N = "
        assert gui.lbl_peak_value.text() == "Profundidad: --"
        monitor_labels = {label.text() for label in gui._tabs.widget(1).findChildren(QLabel)}
        assert "Paso" in monitor_labels
        assert "Paso (µm)" not in monitor_labels
        assert {"Min", "Max"}.issubset(monitor_labels)
        assert "Min (mm)" not in monitor_labels
        assert "Max (mm)" not in monitor_labels
        assert "Res (µm)" not in monitor_labels
        assert all(sp.suffix() == " mm" for sp in gui.window_depth_min_spinboxes)
        assert all(sp.suffix() == " mm" for sp in gui.window_depth_max_spinboxes)
        assert all(sp.suffix() == " µm" for sp in gui.window_resolution_spinboxes)
        assert gui.spin_global_depth_min_mm.suffix() == " mm"
        assert gui.spin_global_resolution_um.suffix() == " µm"
        normalized_analysis_boxes = (
            list(gui.window_depth_min_spinboxes)
            + list(gui.window_depth_max_spinboxes)
            + list(gui.window_resolution_spinboxes)
            + [
                gui.spin_global_depth_min_mm,
                gui.spin_global_resolution_um,
                gui.spin_n_peaks,
                gui.fiber_diameter_spinbox,
                gui.wavelength_central_spinbox,
                gui.collimator_focal_length_spinbox,
                gui.objective_focal_length_spinbox,
            ]
        )
        assert {spin.width() for spin in normalized_analysis_boxes} == {112}
        assert all(
            spin.toolTip() == (
                "Paso axial en µm.\n"
                "Determina el espaciado entre puntos en el eje Z\n"
                "dentro de esta ventana."
            )
            for spin in gui.window_resolution_spinboxes
        )
        assert gui.spin_global_resolution_um.toolTip() == (
            "Paso axial global en µm.\n"
            "Determina el espaciado entre puntos en el eje Z\n"
            "cuando no hay ventanas activas."
        )
        monitor_group_boxes = gui._tabs.widget(1).findChildren(QGroupBox)
        assert all(group.styleSheet() == gui._driver_group_style() for group in monitor_group_boxes)
        barrido_widget = gui._tabs.widget(2)
        barrido_groups = [group.title() for group in barrido_widget.findChildren(QGroupBox)]
        assert barrido_groups == [
            "Muestra",
            "Adquisición por punto",
            "Barrido",
            "Guardar",
        ]
        barrido_section_labels = {
            label.text()
            for label in barrido_widget.findChildren(QLabel)
            if label.text() in {
                "IDENTIFICACIÓN",
                "ADQUISICIÓN",
                "EJES",
                "RECORRIDO",
                "GUARDADO",
            }
        }
        assert barrido_section_labels == {
            "IDENTIFICACIÓN",
            "ADQUISICIÓN",
            "EJES",
            "RECORRIDO",
            "GUARDADO",
        }
        barrido_labels = {label.text() for label in barrido_widget.findChildren(QLabel)}
        assert {"Inicio", "Fin", "Paso", "Espera"}.issubset(barrido_labels)
        assert gui.spin_settling_time_s.suffix() == " s"
        assert all(
            spin.suffix() == " mm"
            for axis_spinboxes in gui.scan_spinboxes.values()
            for spin in axis_spinboxes
        )
        assert gui.spin_settling_time_s.maximum() == pytest.approx(9999.999)
        assert gui.spin_settling_time_s.value() == pytest.approx(0.050)
        assert gui.spin_settling_time_s.minimumWidth() == 0
        assert not hasattr(gui, "spin_settling_time_ms")
        assert not hasattr(gui, "spin_global_res")
        assert hasattr(gui, "_wavelengths_cached_nm")
        assert not hasattr(gui, "_wavelengths_cached")
        assert gui._build_scan_config().settling_time_s == pytest.approx(0.050)
        assert set(gui.scan_spinboxes) == {1, 2, 3}
        gui._tabs.setCurrentIndex(2)
        qapp.processEvents()
        assert gui.spin_measurements_per_point.width() > gui.spin_settling_time_s.width()
        gui._tabs.setCurrentIndex(0)
        qapp.processEvents()
        barrido_group_boxes = barrido_widget.findChildren(QGroupBox)
        assert all(group.styleSheet() == gui._driver_group_style() for group in barrido_group_boxes)
        assert gui._tabs.indexOf(gui.spin_exposure_ms.parentWidget().parentWidget()) == 0
        assert [checkbox.text() for checkbox in gui.window_enabled_checkboxes] == [
            "V1", "V2", "V3", "V4", "V5"
        ]
        assert not hasattr(gui, "lbl_opt_res")
        driver_labels = {
            label.text() for label in gui._tabs.widget(0).findChildren(QLabel)
        }
        assert {
            "VISUALIZACIÓN",
            "CONEXIÓN",
            "ESTADO REAL",
            "MOVIMIENTO MANUAL",
            "Detector:",
            "Exposición",
            "Tolerancia:",
            "Estado: IDLE",
        }.issubset(driver_labels)
        assert "Exposición (ms)" not in driver_labels
        assert "Tolerancia (µm):" not in driver_labels
        assert "Dispositivo:" not in driver_labels
        assert "HOME / HABILITACIÓN" not in driver_labels
        assert "POSICIÓN REAL" not in driver_labels
        driver_widget = gui._tabs.widget(0)
        assert driver_widget.findChildren(QLabel)
        assert gui.combo_spectrometer.y() < gui.spin_exposure_ms.y()
        assert gui.lbl_resolution.y() < gui.lbl_axial_range.y() < gui.lbl_system_state.y()
        assert gui.combo_motion_controller.y() < gui.btn_motion_stop.y()
        assert gui.btn_motion_stop.y() < gui.lbl_pos_actual.y()
        assert gui.lbl_pos_actual.y() < gui.manual_pos[1].mapTo(
            gui._tabs.widget(0), gui.manual_pos[1].rect().topLeft()
        ).y()
        for axis in (1, 2, 3):
            assert gui.motor_position_labels[axis].x() == gui.motor_toggle_buttons[axis].x()
            assert gui.motor_position_labels[axis].y() > gui.motor_toggle_buttons[axis].y()
        assert gui.lbl_resolution.text() == "Resolución axial: -- µm"
        assert gui.lbl_axial_range.text() == "Rango axial: -- mm"
        assert gui.btn_start_monitor.text() == "▶ Iniciar"
        assert "adquisición" not in gui.btn_start_monitor.toolTip().lower()
        assert "visualización continua" in gui.btn_start_monitor.toolTip().lower()
        assert "MONITOR = visualización en tiempo real." in gui.lbl_system_state.toolTip()
        assert "MONITOR = adquisición" not in gui.lbl_system_state.toolTip()
        assert gui.spin_exposure_ms.toolTip() == (
            "Tiempo de integración del espectrómetro en milisegundos."
        )
        wavelengths = np.linspace(780.0, 920.0, 3648)
        gui._update_spectrometer_metrics(wavelengths)
        assert gui.lbl_resolution.text() == "Resolución axial: 2.28 µm"
        assert gui.lbl_axial_range.text() == "Rango axial: 4.67 mm"
        assert gui.lbl_global_depth_max.text() == "→ 4.67 mm"
        assert gui.spin_global_depth_min_mm.maximum() == pytest.approx(4.67337, abs=1e-3)
        config = gui._build_processing_config()
        assert config.depth_max_global_m == pytest.approx(4.67337e-3)
        assert ProcessingEngine.theoretical_depth_range_mm(780.0, 920.0, 3648) == pytest.approx(4.67337)
        assert gui.btn_stop_monitor.text() == "■ Detener"
        assert gui.btn_start_monitor.width() == gui.btn_stop_monitor.width()
        assert gui.btn_stop_monitor.x() > gui.btn_start_monitor.x()
        assert gui.btn_motor_reconnect.text() == "Reconectar"
        assert gui.lbl_spec_status.text() == "Conectado · MockSpectrometer"
        assert gui.lbl_motor_status.text() == "Conectado · MockMotionController"
        assert gui.btn_spec_reconnect.x() == gui.combo_spectrometer.x()
        assert gui.btn_spec_reconnect.y() > gui.combo_spectrometer.y()
        assert gui.btn_spec_reconnect.width() == gui.combo_spectrometer.width()
        driver_group_boxes = gui._tabs.widget(0).findChildren(QGroupBox)
        assert all(not group.toolTip() for group in driver_group_boxes)
        assert not gui.btn_stop_monitor.testAttribute(Qt.WA_AlwaysShowToolTips)
        assert not hasattr(gui, "lbl_opt_zr")
        optics_labels = [
            label.text() for label in gui._tabs.widget(1).findChildren(QGroupBox)[2].findChildren(QLabel)
        ]
        assert all("Spot difr." not in text and "Rayleigh" not in text for text in optics_labels)
        assert {"Ø_f", "λ", "f_col", "f_obj"}.issubset(set(optics_labels))
        assert not any(
            text in {"Ø_f (µm)", "λ (nm)", "f_col (mm)", "f_obj (mm)"}
            for text in optics_labels
        )
        assert gui.fiber_diameter_spinbox.suffix() == " µm"
        assert gui.wavelength_central_spinbox.suffix() == " nm"
        assert gui.collimator_focal_length_spinbox.suffix() == " mm"
        assert gui.objective_focal_length_spinbox.suffix() == " mm"
        assert any(text.startswith("Ø_col:") for text in optics_labels)
        assert any(text.startswith("NA:") for text in optics_labels)
        assert any(text.startswith("Ø_focal:") for text in optics_labels)
        assert not any(text.startswith("Ø_spot:") for text in optics_labels)
        assert gui.lbl_opt_geo.toolTip() == (
            "Diámetro de cintura gaussiana en el plano focal (µm).\n"
            "Se calcula propagando W0 hasta Wz y resolviendo la cintura Ws.\n"
            "Ø_focal = 2 · Ws"
        )
        assert "Spot por imagen geométrica de la fibra" not in gui.lbl_opt_geo.toolTip()
        assert any(text.startswith("b_conf:") for text in optics_labels)
        assert gui.lbl_opt_beam.y() == gui.lbl_opt_geo.y()
        assert gui.lbl_opt_na.y() == gui.lbl_opt_confocal.y()
        assert gui.lbl_opt_beam.x() == gui.lbl_opt_na.x()
        assert gui.lbl_opt_geo.x() == gui.lbl_opt_confocal.x()
        assert gui.lbl_opt_beam.width() == gui.lbl_opt_geo.width()
        assert gui.lbl_opt_na.width() == gui.lbl_opt_confocal.width()
        assert [
            gui.motor_toggle_buttons[axis].text() for axis in (1, 2, 3)
        ] == ["Habilitar X", "Habilitar Y", "Habilitar Z"]
        assert [
            gui.motor_status_labels[axis].text() for axis in (1, 2, 3)
        ] == ["● OFF", "● OFF", "● OFF"]
        assert gui.spin_position_tolerance_um.value() == pytest.approx(2.0)
        assert gui.spin_position_tolerance_um.minimum() == pytest.approx(0.5)
        assert gui.spin_position_tolerance_um.suffix() == " µm"
        assert all(sp.suffix() == " mm" for sp in gui.manual_pos.values())
        manual_buttons = {
            button.text(): button
            for button in gui._manual_move_buttons
            if button.text().startswith("Mover ")
        }
        assert set(manual_buttons) == {"Mover X", "Mover Y", "Mover Z"}
        for axis, name in ((1, "X"), (2, "Y"), (3, "Z")):
            spin = gui.manual_pos[axis]
            button = manual_buttons[f"Mover {name}"]
            assert spin.parentWidget() is button.parentWidget()
            assert spin.geometry().right() < button.geometry().left()
            assert spin.geometry().y() == button.geometry().y()
            assert spin.width() > button.width()
        assert gui.lbl_pos_actual.text().startswith("Posición real:")
        assert gui._motor_position_timer.isActive()
        gui.manual_pos[1].setValue(1.2345)
        gui.manual_pos[2].setValue(-4.25)
        gui._on_manual_move_finished([(2, -5.5)])
        assert gui.manual_pos[1].value() == pytest.approx(1.2345)
        assert gui.manual_pos[2].value() == pytest.approx(-4.25)
        gui._sync_motor_positions()
        assert gui.lbl_pos_actual.text() == "Posición real:"
        assert [gui.motor_position_labels[axis].text() for axis in (1, 2, 3)] == [
            "X=0.0000 mm",
            "Y=0.0000 mm",
            "Z=0.0000 mm",
        ]
        assert all(
            label.fontMetrics().horizontalAdvance(label.text()) <= label.contentsRect().width()
            for label in gui.motor_position_labels.values()
        )
        assert gui.manual_pos[1].value() == pytest.approx(1.2345)
        assert gui.manual_pos[2].value() == pytest.approx(-4.25)
        gui.spin_position_tolerance_um.setValue(0.5)
        assert gui.spin_position_tolerance_um.value() == pytest.approx(0.5)
        assert gui.spin_exposure_ms.suffix() == " ms"

        gui._enable_motor(1)
        assert gui.motor_toggle_buttons[1].text() == "Deshabilitar X"
        assert gui.motor_status_labels[1].text() == "● ON"
        gui._disable_motor(1)
        assert gui.motor_toggle_buttons[1].text() == "Habilitar X"
        assert gui.motor_status_labels[1].text() == "● OFF"

        class FakeSpectrometer:
            def __init__(self):
                self.dark_enabled = False
                self.nonlinearity_enabled = False
                self.exposure_values = []

            def capabilities(self):
                return {
                    "exposure_control": True,
                    "dark_correction": True,
                    "nonlinearity_correction": True,
                    "wavelengths_nm": True,
                    "reconnect": True,
                }

            def set_exposure_ms(self, value):
                self.exposure_values.append(float(value))

            def set_dark_correction(self, enabled):
                self.dark_enabled = bool(enabled)

            def set_nonlinearity_correction(self, enabled):
                self.nonlinearity_enabled = bool(enabled)

        fake_spec = FakeSpectrometer()
        gui.spec = fake_spec
        gui._set_state(SystemState.MONITORING)

        gui.chk_dark.setChecked(True)
        assert fake_spec.dark_enabled is True
        gui.chk_dark.setChecked(False)
        assert fake_spec.dark_enabled is False

        gui.chk_nonlinearity.setChecked(True)
        assert fake_spec.nonlinearity_enabled is True
        gui.chk_nonlinearity.setChecked(False)
        assert fake_spec.nonlinearity_enabled is False

        gui.spin_exposure_ms.setValue(25.0)
        assert fake_spec.exposure_values[-1] == 25.0
    finally:
        gui.close()


def test_numeric_suffixes_fit_all_visible_tabs(qapp):
    gui = OCTGUI(use_mock=True)
    gui.show()
    qapp.processEvents()
    try:
        for index in range(gui._tabs.count()):
            gui._tabs.setCurrentIndex(index)
            qapp.processEvents()
            assert gui._tabs.currentWidget().width() <= 416
            for spin in gui._tabs.currentWidget().findChildren(QDoubleSpinBox):
                if not spin.suffix():
                    continue
                required = QFontMetrics(spin.lineEdit().font()).horizontalAdvance(
                    spin.text()
                )
                available = spin.lineEdit().contentsRect().width()
                assert required <= available, (
                    gui._tabs.tabText(index),
                    spin.text(),
                    available,
                    required,
                )

        gui._tabs.setCurrentIndex(1)
        qapp.processEvents()
        for spin_min, spin_max, spin_step in zip(
            gui.window_depth_min_spinboxes,
            gui.window_depth_max_spinboxes,
            gui.window_resolution_spinboxes,
        ):
            assert spin_min.parentWidget() is spin_max.parentWidget() is spin_step.parentWidget()
            assert spin_min.geometry().right() < spin_max.geometry().left()
            assert spin_max.geometry().right() < spin_step.geometry().left()

        gui._tabs.setCurrentIndex(2)
        qapp.processEvents()
        for spin_start, spin_end, spin_step in gui.scan_spinboxes.values():
            assert spin_start.parentWidget() is spin_end.parentWidget() is spin_step.parentWidget()
            assert spin_start.geometry().right() < spin_end.geometry().left()
            assert spin_end.geometry().right() < spin_step.geometry().left()
    finally:
        gui.close()


def test_plot_color_reservation_for_spectrum_and_global_profile(qapp):
    gui = OCTGUI(use_mock=True)
    try:
        spectrum_curve = gui.plot_spec.plot(
            [800.0, 850.0], [1.0, 2.0], clear=True, pen=SPECTRUM_COLOR
        )
        spectrum_color = spectrum_curve.curve.opts["pen"].color().name()
        assert GLOBAL_PROFILE_COLOR == SPECTRUM_COLOR
        assert SPECTRUM_COLOR not in WINDOW_COLORS

        profile = SimpleNamespace(
            depth_axis_m=np.array([0.0, 1e-3]),
            profile=np.array([1.0, 2.0]),
        )
        global_result = SimpleNamespace(
            measurements_per_point=1,
            is_windowed=False,
            window_results=[{0: profile}],
            peaks=[{}],
        )
        gui._plot_result(global_result)
        assert gui._profile_curves[0].curve.opts["pen"].color().name() == spectrum_color

        windowed_result = SimpleNamespace(
            measurements_per_point=1,
            is_windowed=True,
            window_results=[{0: profile}],
            peaks=[{}],
        )
        gui._plot_result(windowed_result)
        assert gui._profile_curves[0].curve.opts["pen"].color().name() != spectrum_color
    finally:
        gui.close()


def test_gui_state_transitions_preserve_capability_and_monitor_locks(qapp):
    gui = OCTGUI(use_mock=True)
    try:
        for state in (
            SystemState.IDLE,
            SystemState.MONITORING,
            SystemState.SCANNING,
            SystemState.ERROR,
        ):
            gui._set_state(state)
            assert gui.chk_nonlinearity.isEnabled() is False
            assert gui.btn_start_monitor.isEnabled() is (
                state in (SystemState.IDLE, SystemState.ERROR)
            )
            assert gui.btn_stop_monitor.isEnabled() is (
                state == SystemState.MONITORING
            )
            assert gui._manual_move_buttons[0].isEnabled() is (
                state != SystemState.SCANNING
            )
            assert gui.motor_toggle_buttons[1].isEnabled() is (
                state != SystemState.SCANNING
            )
    finally:
        gui.close()


def test_scan_request_stops_monitor_before_starting(qapp, monkeypatch):
    gui = OCTGUI(use_mock=True)
    try:
        gui._set_state(SystemState.MONITORING)
        assert gui.btn_run_scan.isEnabled()

        stop_calls = []

        def fake_stop_monitor():
            stop_calls.append(True)
            gui._set_state(SystemState.IDLE)

        gui.stop_monitor = fake_stop_monitor
        monkeypatch.setattr(
            "gui.main_gui.QMessageBox.warning",
            lambda *args, **kwargs: None,
        )
        gui.run_scan()

        assert stop_calls == [True]
        assert gui._state == SystemState.IDLE
    finally:
        gui.close()


def test_monitor_stop_before_thread_start_is_sticky(qapp):
    worker = MonitorWorker(None, None)
    worker.stop()
    worker.start()
    assert worker.wait(1000)
