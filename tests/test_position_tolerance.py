import pytest

from constants import DEFAULT_POSITION_TOLERANCE_UM, MIN_POSITION_TOLERANCE_UM
from gui.workers import MoveWorker, ScanWorker


def test_position_tolerance_defaults_to_two_micrometers_and_allows_half():
    assert DEFAULT_POSITION_TOLERANCE_UM == pytest.approx(2.0)
    assert MIN_POSITION_TOLERANCE_UM == pytest.approx(0.5)
    assert DEFAULT_POSITION_TOLERANCE_UM * 1e-3 == pytest.approx(0.002)
    assert MIN_POSITION_TOLERANCE_UM * 1e-3 == pytest.approx(0.0005)


def test_move_worker_passes_explicit_position_tolerance():
    class Motor:
        def __init__(self):
            self.calls = []

        def goto_and_wait(self, axis, target, tolerance_mm):
            self.calls.append((axis, target, tolerance_mm))

        def get_position(self, axis):
            return 0.0

    motor = Motor()
    worker = MoveWorker(motor, [(1, 0.5)], position_tolerance_mm=0.0005)
    worker.run()

    assert motor.calls == [(1, 0.5, pytest.approx(0.0005))]


def test_scan_worker_passes_explicit_position_tolerance():
    class Config:
        active_axes = ["X"]
        settling_time_s = 0.0

        def _axis_params(self, axis_name):
            if axis_name == "X":
                return True, 0.0, 0.5, 0.5
            return False, 0.0, 0.0, 0.1

        def iter_points(self, inactive_pos=None):
            yield (0.5, 0.0, 0.0)

    class Motor:
        def __init__(self):
            self.calls = []
            self.positions = {1: 0.0, 2: 0.0, 3: 0.0}

        def get_position(self, axis):
            return self.positions[axis]

        def goto_and_wait(self, axis, target, tolerance_mm):
            self.calls.append((axis, target, tolerance_mm))
            self.positions[axis] = target

    class Acquisition:
        def acquire_point(self, *args):
            return object()

    class Processing:
        def process(self, raw):
            return object()

    motor = Motor()
    worker = ScanWorker(
        motor,
        Acquisition(),
        Processing(),
        Config(),
        1,
        position_tolerance_mm=0.0005,
    )
    worker.run()

    assert motor.calls[0] == (1, 0.5, pytest.approx(0.0005))
    assert motor.calls[-1] == (1, 0.0, pytest.approx(0.0005))
    assert all(call[2] == pytest.approx(0.0005) for call in motor.calls)