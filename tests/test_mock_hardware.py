import threading
import time

import numpy as np
import pytest

from hardware.motor_mock import MockMotionController
from hardware.spectrometer_mock import MockSpectrometer


def test_mock_motor_stop_interrupts_active_motion_before_target():
    motor = MockMotionController(motion_speed_mm_s=5.0)
    motor.connect()
    errors = []

    def move():
        try:
            motor.goto_and_wait(1, 10.0)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=move)
    thread.start()

    deadline = time.monotonic() + 1.0
    while motor.get_position(1) <= 0.0 and time.monotonic() < deadline:
        time.sleep(0.005)

    motor.stop_motion(1)
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], motor.MotionCancelled)
    assert 0.0 < motor.get_position(1) < 10.0


def test_mock_motor_reaches_target_when_not_stopped():
    motor = MockMotionController(motion_speed_mm_s=1000.0)
    motor.connect()

    motor.goto_and_wait(1, 2.0)

    assert motor.get_position(1) == pytest.approx(2.0)


def test_mock_dark_correction_subtracts_configured_offset():
    spectrometer = MockSpectrometer(
        n_pixels=32,
        noise_level=0.0,
        dark_counts=123.0,
    )
    spectrometer.connect()

    spectrometer.set_dark_correction(False)
    _, raw = spectrometer.read()

    spectrometer.set_dark_correction(True)
    _, corrected = spectrometer.read()

    np.testing.assert_allclose(raw - corrected, 123.0)


def test_mock_dark_correction_accepts_per_pixel_offset():
    dark = np.linspace(0.0, 31.0, 32)
    spectrometer = MockSpectrometer(
        n_pixels=32,
        noise_level=0.0,
        dark_counts=dark,
    )
    spectrometer.connect()

    spectrometer.set_dark_correction(False)
    _, raw = spectrometer.read()

    spectrometer.set_dark_correction(True)
    _, corrected = spectrometer.read()

    np.testing.assert_allclose(raw - corrected, dark)
