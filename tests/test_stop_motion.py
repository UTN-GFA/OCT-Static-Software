import pytest

from hardware.interfaces import MotionControllerInterface
from hardware.motor_mock import MockMotionController
from hardware.motor_esp301 import ESP301MotionController
from gui.workers import MoveWorker


def test_motion_interface_exposes_stop_motion():
    assert hasattr(MotionControllerInterface, "stop_motion")


def test_mock_stop_motion_records_cancellation_and_raises_on_active_move():
    motor = MockMotionController()
    motor.connect()
    motor.stop_motion(1)
    assert motor.stop_requests == [1]

    with pytest.raises(motor.MotionCancelled):
        motor.goto_and_wait(1, 30.0)


def test_mock_stop_is_safe_for_disconnected_controller():
    motor = MockMotionController()
    with pytest.raises(RuntimeError, match="no conectado"):
        motor.stop_motion(1)


def test_move_worker_reports_stopped_instead_of_error():
    motor = MockMotionController()
    motor.connect()
    worker = MoveWorker(motor, [(1, 30.0)], position_tolerance_mm=0.0005)
    stopped = []
    errors = []
    worker.stopped.connect(lambda: stopped.append(True))
    worker.error_occurred.connect(errors.append)

    worker.stop()
    worker.run()

    assert stopped == [True]
    assert errors == []


def test_esp301_stop_motion_sends_axis_stop_command():
    class FakeSerial:
        is_open = True
        in_waiting = 0

        def __init__(self):
            self.writes = []

        def write(self, payload):
            self.writes.append(payload)

        def readline(self):
            return b""

    motor = ESP301MotionController()
    motor._ser = FakeSerial()
    motor._available_axes = {1}
    motor.stop_motion(1)

    assert motor._ser.writes == [b"1ST\r"]
    assert motor._stop_requested == {1}


def test_esp301_stop_motion_does_not_read_with_long_wait_timeout():
    class StopSerial:
        is_open = True
        in_waiting = 0
        timeout = 30.0

        def __init__(self):
            self.writes = []
            self.readline_calls = 0

        def write(self, payload):
            self.writes.append(payload)

        def readline(self):
            self.readline_calls += 1
            raise AssertionError("STOP no debe esperar una respuesta serial")

    motor = ESP301MotionController()
    motor._ser = StopSerial()
    motor._available_axes = {1}

    motor.stop_motion(1)

    assert motor._ser.writes == [b"1ST\r"]
    assert motor._ser.readline_calls == 0
