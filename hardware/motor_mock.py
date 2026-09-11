# -*- coding: utf-8 -*-
"""
hardware/motor_mock.py
Controlador de motores simulado para desarrollo y testing.
"""

import logging
import time
from .interfaces import MotionCancelled, MotionControllerInterface
from constants import AXIS_NAMES, DEFAULT_POSITION_TOLERANCE_MM

logger = logging.getLogger(__name__)


class MockMotionController(MotionControllerInterface):
    MotionCancelled = MotionCancelled
    """
    Controlador de motores simulado.
    Movimientos progresivos simulados, posiciones en memoria.

    Parámetros
    ----------
    axes : set
        Ejes disponibles (default: {1, 2, 3} = X, Y, Z).
    initial_position : float
        Posición inicial de todos los ejes en mm.
    motion_speed_mm_s : float
        Velocidad simulada en mm/s.
    """

    def __init__(
        self,
        axes: set = None,
        initial_position: float = 0.0,
        motion_speed_mm_s: float = 100.0,
        port: str = None,
    ):
        if motion_speed_mm_s <= 0:
            raise ValueError("motion_speed_mm_s debe ser mayor que cero")
        self._axes = axes or {1, 2, 3}
        self._positions = {ax: initial_position for ax in self._axes}
        self._motion_speed_mm_s = float(motion_speed_mm_s)
        self._enabled = {ax: False for ax in self._axes}
        self._connected = False
        self._stop_requested = set()
        self.stop_requests = []

    def capabilities(self) -> dict:
        return {
            "absolute_move": True,
            "relative_move": False,
            "position_readback": True,
            "stop_motion": True,
            "axis_enable": True,
            "homing": False,
            "velocity_control": False,
            "limits": False,
        }

    def connect(self) -> bool:
        self._connected = True
        detected = [AXIS_NAMES.get(a, str(a))
                    for a in sorted(self._axes)]
        logger.info("MockMotionController conectado, ejes: %s", detected)
        return True

    def move_absolute(self, axis: int, position_mm: float) -> None:
        self._check_axis(axis)
        self._positions[axis] = position_mm

    def get_position(self, axis: int) -> float:
        self._check_axis(axis)
        return self._positions[axis]

    def goto_and_wait(
        self,
        axis: int,
        position_mm: float,
        tolerance_mm: float = DEFAULT_POSITION_TOLERANCE_MM,
        timeout_s: float = 30.0,
    ) -> float:
        self._check_axis(axis)
        if axis in self._stop_requested:
            self._stop_requested.remove(axis)
            raise self.MotionCancelled(f"Movimiento del eje {axis} detenido")
        start_position = self._positions[axis]
        distance = position_mm - start_position
        if abs(distance) <= tolerance_mm:
            self._positions[axis] = position_mm
            return self._positions[axis]

        started = time.monotonic()
        while True:
            if axis in self._stop_requested:
                self._stop_requested.remove(axis)
                raise self.MotionCancelled(f"Movimiento del eje {axis} detenido")

            elapsed = time.monotonic() - started
            travelled = min(abs(distance), elapsed * self._motion_speed_mm_s)
            fraction = travelled / abs(distance)
            self._positions[axis] = start_position + distance * fraction
            if travelled >= abs(distance) - tolerance_mm:
                self._positions[axis] = position_mm
                break

            if elapsed >= timeout_s:
                raise TimeoutError(f"Timeout moviendo eje {axis}")
            time.sleep(0.005)
        logger.debug("MockMotor eje %d → %.4f mm", axis, position_mm)
        return self._positions[axis]

    def stop_motion(self, axis: int) -> None:
        self._check_axis(axis)
        self._stop_requested.add(axis)
        self.stop_requests.append(axis)

    def enable_axis(self, axis: int) -> None:
        self._check_axis(axis)
        self._enabled[axis] = True

    def disable_axis(self, axis: int) -> None:
        self._check_axis(axis)
        self._enabled[axis] = False

    def close(self) -> None:
        self._connected = False
        logger.info("MockMotionController desconectado")

    @property
    def available_axes(self) -> set:
        return self._axes.copy()

    def _check_axis(self, axis: int) -> None:
        if not self._connected:
            raise RuntimeError("MockMotionController: no conectado")
        if axis not in self._axes:
            raise RuntimeError(
                f"MockMotionController: eje {axis} no disponible "
                f"(disponibles: {self._axes})")
