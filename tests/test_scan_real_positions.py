import pytest

from hardware.motor_esp301 import ESP301MotionController
from gui.workers import ScanWorker
from PySide6.QtCore import QCoreApplication


def test_esp301_goto_and_wait_returns_verified_position(monkeypatch):
    motor = ESP301MotionController()
    motor._ser = type("Serial", (), {"is_open": True})()
    motor._available_axes = {1}
    monkeypatch.setattr(motor, "_send", lambda command: "")
    monkeypatch.setattr(motor, "_send_wait", lambda command, timeout_s: None)
    monkeypatch.setattr(motor, "get_position", lambda axis: 0.4996)

    actual = motor.goto_and_wait(1, 0.5, tolerance_mm=0.001)

    assert actual == pytest.approx(0.4996)


def test_scan_worker_passes_verified_coordinates_to_acquisition_and_signal():
    app = QCoreApplication.instance() or QCoreApplication([])

    class Config:
        active_axes = ["X"]
        settling_time_s = 0.0

        def _axis_params(self, axis_name):
            if axis_name == "X":
                return True, 0.0, 0.5, 0.5
            return False, 0.0, 0.0, 0.1

        def iter_points(self, inactive_pos=None):
            yield (0.5, 0.0, 0.0)

    class DriftMotion:
        def __init__(self):
            self.positions = {1: 0.0, 2: 0.0, 3: 0.0}

        def get_position(self, axis):
            if axis == 1 and self.positions[axis] == 0.4996:
                return 0.4997
            return self.positions[axis]

        def goto_and_wait(self, axis, target, tolerance_mm):
            assert tolerance_mm == pytest.approx(0.0005)
            actual = 0.4996 if axis == 1 and target == 0.5 else target
            self.positions[axis] = actual
            return actual

    class Acquisition:
        def __init__(self):
            self.calls = []

        def acquire_point(self, x_mm, y_mm, z_mm, measurements):
            self.calls.append((x_mm, y_mm, z_mm, measurements))
            return object()

    class Processing:
        def process(self, raw):
            return object()

    acq = Acquisition()
    worker = ScanWorker(
        DriftMotion(), acq, Processing(), Config(), 1, position_tolerance_mm=0.0005
    )
    points = []
    worker.point_ready.connect(lambda *args: points.append(args[:3]))

    worker.run()

    assert acq.calls == [(pytest.approx(0.4997), 0.0, 0.0, 1)]
    assert points == [(pytest.approx(0.4997), 0.0, 0.0)]
    assert app is not None