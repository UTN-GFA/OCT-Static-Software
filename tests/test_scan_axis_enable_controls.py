from types import SimpleNamespace

import pytest
from PyQt5.QtWidgets import QApplication, QCheckBox, QMainWindow

from gui.main_gui import OCTGUI
from gui.workers import SystemState


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _gui_with_axis_checks():
    gui = OCTGUI.__new__(OCTGUI)
    QMainWindow.__init__(gui)
    gui._state = SystemState.IDLE
    gui._motor_enabled = {1: False, 2: False, 3: False}
    gui.mot = SimpleNamespace(available_axes={1, 2, 3})
    gui.scan_checks = {axis: QCheckBox(name) for axis, name in ((1, "X"), (2, "Y"), (3, "Z"))}
    return gui


def test_scan_axis_checks_follow_per_axis_motor_enable_state(qapp):
    gui = _gui_with_axis_checks()

    gui._update_scan_axis_checks()
    assert [gui.scan_checks[axis].isEnabled() for axis in (1, 2, 3)] == [False, False, False]

    gui._motor_enabled[1] = True
    gui._update_scan_axis_checks()
    assert [gui.scan_checks[axis].isEnabled() for axis in (1, 2, 3)] == [True, False, False]

    gui.scan_checks[1].setChecked(True)
    gui._motor_enabled[1] = False
    gui._update_scan_axis_checks()
    assert gui.scan_checks[1].isChecked() is False
    assert [gui.scan_checks[axis].isEnabled() for axis in (1, 2, 3)] == [False, False, False]


def test_scan_axis_checks_are_blocked_during_scan_even_if_motor_enabled(qapp):
    gui = _gui_with_axis_checks()
    gui._motor_enabled = {1: True, 2: True, 3: False}
    gui._state = SystemState.SCANNING

    gui._update_scan_axis_checks()

    assert [gui.scan_checks[axis].isEnabled() for axis in (1, 2, 3)] == [False, False, False]


def test_enable_and_disable_buttons_update_the_matching_scan_check(qapp):
    gui = _gui_with_axis_checks()

    class FakeMotion:
        available_axes = {1, 2, 3}

        def enable_axis(self, axis):
            self.enabled_axis = axis

        def disable_axis(self, axis):
            self.disabled_axis = axis

    gui.mot = FakeMotion()

    gui._enable_motor(2)
    assert gui._motor_enabled[2] is True
    assert gui.scan_checks[2].isEnabled() is True
    assert gui.scan_checks[1].isEnabled() is False

    gui.scan_checks[2].setChecked(True)
    gui._disable_motor(2)
    assert gui._motor_enabled[2] is False
    assert gui.scan_checks[2].isChecked() is False
    assert gui.scan_checks[2].isEnabled() is False
