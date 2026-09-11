import pytest
from PyQt5.QtWidgets import QApplication

from gui.main_gui import OCTGUI
from hardware import (
    create_motion_controller,
    create_spectrometer,
    list_motion_controllers,
    list_spectrometers,
)
from hardware.motor_mock import MockMotionController
from hardware.spectrometer_mock import MockSpectrometer


class _RecordingSpectrometer:
    def __init__(self, capabilities):
        self._capabilities = capabilities
        self.calls = []

    def capabilities(self):
        return dict(self._capabilities)

    def set_exposure_ms(self, value):
        self.calls.append(("exposure_control", value))

    def set_dark_correction(self, enabled):
        self.calls.append(("dark_correction", bool(enabled)))

    def set_nonlinearity_correction(self, enabled):
        self.calls.append(("nonlinearity_correction", bool(enabled)))


class _CapabilityMotor:
    def __init__(self, capabilities):
        self._capabilities = capabilities
        self.available_axes = {1}

    def capabilities(self):
        return dict(self._capabilities)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_registry_exposes_user_facing_device_options_without_importing_gui():
    assert ("hr4000", "Ocean Optics HR4000") in list_spectrometers()
    assert ("mock", "Espectrómetro simulado") in list_spectrometers()
    assert ("esp301", "Newport ESP301") in list_motion_controllers()
    assert ("mock", "Controlador de motores simulado") in list_motion_controllers()


def test_registered_drivers_declare_capabilities():
    spectrometer_caps = create_spectrometer("mock").capabilities()
    motor_caps = create_motion_controller("mock").capabilities()

    assert spectrometer_caps["exposure_control"] is True
    assert spectrometer_caps["dark_correction"] is True
    assert spectrometer_caps["nonlinearity_correction"] is False
    assert spectrometer_caps["wavelengths_nm"] is True

    assert motor_caps["absolute_move"] is True
    assert motor_caps["relative_move"] is False
    assert motor_caps["position_readback"] is True
    assert motor_caps["stop_motion"] is True
    assert motor_caps["axis_enable"] is True
    assert motor_caps["homing"] is False


def test_capabilities_are_available_before_connecting():
    assert isinstance(MockSpectrometer().capabilities(), dict)
    assert isinstance(MockMotionController().capabilities(), dict)


def test_factory_passes_common_motor_configuration_to_registered_drivers():
    motor = create_motion_controller("mock", port="COM3")
    assert motor is not None


@pytest.mark.parametrize(
    "factory, kind",
    [
        (create_spectrometer, "mock"),
        (create_motion_controller, "mock"),
    ],
)
def test_factory_keeps_registry_selection_contract(factory, kind):
    assert factory(kind) is not None


def test_gui_populates_hardware_selectors_from_registry(qapp):
    gui = OCTGUI(use_mock=True)
    try:
        assert gui.combo_spectrometer.count() == len(list_spectrometers())
        assert gui.combo_motion_controller.count() == len(list_motion_controllers())
        assert gui.combo_spectrometer.currentData() == "mock"
        assert gui.combo_motion_controller.currentData() == "mock"
        assert gui.chk_dark.isEnabled() is True
        assert gui.chk_nonlinearity.isEnabled() is False
        assert gui.spin_exposure_ms.isEnabled() is True
    finally:
        gui.close()


def test_gui_applies_only_supported_spectrometer_capabilities(qapp):
    caps = {
        "exposure_control": False,
        "dark_correction": False,
        "nonlinearity_correction": False,
        "wavelengths_nm": True,
        "reconnect": False,
    }
    fake = _RecordingSpectrometer(caps)
    gui = OCTGUI(use_mock=True)
    try:
        gui.spec = fake
        gui._use_mock = False
        gui.btn_spec_reconnect.setEnabled(True)
        gui.chk_nonlinearity.setChecked(True)
        gui._apply_hardware_capabilities()

        assert gui.spin_exposure_ms.isEnabled() is False
        assert gui.chk_dark.isEnabled() is False
        assert gui.chk_nonlinearity.isEnabled() is False
        assert gui.chk_nonlinearity.isChecked() is False
        assert gui.btn_spec_reconnect.isEnabled() is False

        gui._apply_spectrometer_settings()
        assert fake.calls == []
    finally:
        gui.close()


def test_gui_applies_all_supported_spectrometer_settings(qapp):
    caps = {
        "exposure_control": True,
        "dark_correction": True,
        "nonlinearity_correction": True,
        "wavelengths_nm": True,
        "reconnect": True,
    }
    fake = _RecordingSpectrometer(caps)
    gui = OCTGUI(use_mock=True)
    try:
        gui.spec = fake
        gui._use_mock = False
        gui._apply_hardware_capabilities()
        gui._apply_spectrometer_settings()

        assert gui.spin_exposure_ms.isEnabled() is True
        assert gui.chk_dark.isEnabled() is True
        assert gui.chk_nonlinearity.isEnabled() is True
        assert {name for name, _ in fake.calls} == {
            "exposure_control",
            "dark_correction",
            "nonlinearity_correction",
        }
    finally:
        gui.close()


def test_gui_disables_operational_controls_without_connected_hardware(qapp):
    gui = OCTGUI(use_mock=True)
    try:
        gui.spec = None
        gui.mot = None
        gui._use_mock = False
        gui._apply_hardware_capabilities()

        assert gui.spin_exposure_ms.isEnabled() is False
        assert gui.chk_dark.isEnabled() is False
        assert gui.chk_nonlinearity.isEnabled() is False
        assert gui.btn_spec_reconnect.isEnabled() is True
        assert all(button.isEnabled() is False for button in gui._manual_move_buttons)
        assert all(button.isEnabled() is False for button in gui.motor_toggle_buttons.values())
    finally:
        gui.close()


def test_gui_blocks_motor_controls_when_capabilities_are_missing(qapp):
    caps = {
        "absolute_move": False,
        "relative_move": False,
        "position_readback": False,
        "stop_motion": False,
        "axis_enable": False,
        "homing": False,
        "velocity_control": False,
        "limits": False,
    }
    gui = OCTGUI(use_mock=True)
    try:
        gui.mot = _CapabilityMotor(caps)
        gui._motor_enabled = {1: True, 2: False, 3: False}
        gui._set_manual_buttons_enabled(True)
        gui._apply_hardware_capabilities()

        assert all(button.isEnabled() is False for button in gui._manual_move_buttons)
        assert all(button.isEnabled() is False for button in gui.motor_toggle_buttons.values())
        assert all(check.isEnabled() is False for check in gui.scan_checks.values())
        assert gui.btn_motion_stop.isEnabled() is False

        caps["absolute_move"] = True
        gui._set_manual_buttons_enabled(True)
        assert all(button.isEnabled() is False for button in gui._manual_move_buttons)
    finally:
        gui.close()


def test_gui_enables_supported_motor_controls_for_available_axis(qapp):
    caps = {
        "absolute_move": True,
        "relative_move": True,
        "position_readback": True,
        "stop_motion": True,
        "axis_enable": True,
        "homing": False,
        "velocity_control": True,
        "limits": False,
    }
    gui = OCTGUI(use_mock=True)
    try:
        gui.mot = _CapabilityMotor(caps)
        gui._motor_enabled = {1: True, 2: False, 3: False}
        gui._set_manual_buttons_enabled(True)
        gui._apply_hardware_capabilities()

        assert all(button.isEnabled() is True for button in gui._manual_move_buttons)
        assert gui.motor_toggle_buttons[1].isEnabled() is True
        assert gui.motor_toggle_buttons[2].isEnabled() is False
        assert gui.scan_checks[1].isEnabled() is True
        assert gui.scan_checks[2].isEnabled() is False
    finally:
        gui.close()
