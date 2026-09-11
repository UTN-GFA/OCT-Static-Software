# -*- coding: utf-8 -*-
"""
hardware/motor_esp301.py
Controlador de motores Newport ESP301 via serial.

Implementa MotionControllerInterface directamente.
"""

import logging
import time
import serial
import numpy as np

from .interfaces import MotionCancelled, MotionControllerInterface
from constants import (
    AXIS_NAMES,
    DEFAULT_MOTOR_BAUD,
    DEFAULT_MOTOR_PORT,
    DEFAULT_MOTOR_SERIAL_TIMEOUT_S,
    DEFAULT_POSITION_TOLERANCE_MM,
)

logger = logging.getLogger(__name__)


class MotorError(Exception):
    """Excepción de motor ESP301."""

    pass


class ESP301MotionController(MotionControllerInterface):
    """
    Control de motores Newport ESP301 - X/Y/Z en mm.

    Parámetros
    ----------
    port : str
        Puerto serial (default: COM3).
    baud : int
        Baudrate (default: 921600).
    """

    def __init__(
        self,
        port: str = DEFAULT_MOTOR_PORT,
        baud: int = DEFAULT_MOTOR_BAUD,
        serial_timeout_s: float = DEFAULT_MOTOR_SERIAL_TIMEOUT_S,
    ):
        self._port = port
        self._baud = baud
        self._serial_timeout_s = serial_timeout_s
        self._ser = None
        self._available_axes: set = set()
        self._stop_requested: set = set()

    # ----------------------------------------------------------
    # MotionControllerInterface
    # ----------------------------------------------------------
    def capabilities(self) -> dict:
        return {
            "absolute_move": True,
            "relative_move": True,
            "position_readback": True,
            "stop_motion": True,
            "axis_enable": True,
            "homing": False,
            "velocity_control": True,
            "limits": False,
        }

    def connect(self) -> bool:
        try:
            self._ser = serial.Serial(
                self._port,
                self._baud,
                timeout=self._serial_timeout_s,
                write_timeout=self._serial_timeout_s,
            )
            time.sleep(0.1)
        except Exception as e:
            logger.error("Error abriendo %s: %s: %s", self._port, type(e).__name__, e)
            return False

        # Detectar ejes disponibles
        self._available_axes = set()
        for axis in (1, 2, 3):
            name = AXIS_NAMES[axis]
            try:
                resp = self._send(f"{axis}TP?")
                resp_clean = resp.strip() if resp else ""
                if resp_clean:
                    try:
                        pos = float(resp_clean)
                        logger.info(
                            "Eje %d (%s): OK pos=%.4f mm", axis, name, pos
                        )
                    except ValueError:
                        logger.info(
                            "Eje %d (%s): resp='%s' (no numérico, incluido)",
                            axis,
                            name,
                            resp_clean,
                        )
                    self._available_axes.add(axis)
                else:
                    logger.info(
                        "Eje %d (%s): sin respuesta — no detectado",
                        axis,
                        name,
                    )
            except Exception as e:
                logger.warning("Eje %d (%s): excepción (%s)", axis, name, e)

        detected = [AXIS_NAMES[a] for a in sorted(self._available_axes)]
        logger.info("ESP301 ejes detectados: %s", detected)
        return True

    def move_absolute(self, axis: int, position_mm: float) -> None:
        self._require_axis(axis)
        self._send(f"{axis}PA{position_mm:.6f}")

    def get_position(self, axis: int) -> float:
        self._require_axis(axis)
        resp = self._send(f"{axis}TP?")
        try:
            p = float(resp)
        except (TypeError, ValueError):
            raise RuntimeError(f"ESP301: posición inválida en eje {axis}: {resp}")
        if not np.isfinite(p):
            raise RuntimeError(f"ESP301: posición inválida en eje {axis}: {p}")
        return p

    def goto_and_wait(
        self,
        axis: int,
        position_mm: float,
        tolerance_mm: float = DEFAULT_POSITION_TOLERANCE_MM,
        timeout_s: float = 30.0,
    ) -> float:
        """
        Movimiento robusto usando WS (Wait for Stop) del ESP301.

        Secuencia:
        1. PA  → mover a posición absoluta.
        2. WS  → el controlador bloquea hasta que el motor frena.
        3. TP? → leer posición final y verificar tolerancia.

        Si la posición no está dentro de tolerancia (atasco mecánico),
        reintenta con jiggle hasta max_attempts veces.

        Raises
        ------
        RuntimeError si falla después de todos los reintentos.
        """
        max_attempts = 3
        self._require_axis(axis)

        for attempt in range(1, max_attempts + 1):
            # Mover
            self._send(f"{axis}PA{position_mm:.6f}")

            # Esperar a que el motor frene (hardware)
            self._send_wait(f"{axis}WS", timeout_s=timeout_s)

            if axis in self._stop_requested:
                self._stop_requested.remove(axis)
                raise MotionCancelled(f"ESP301 eje {axis}: movimiento detenido")

            # Verificar posición final
            try:
                p = self.get_position(axis)
            except RuntimeError:
                if attempt >= max_attempts:
                    raise RuntimeError(f"ESP301 eje {axis}: comunicación perdida")
                continue

            if abs(p - position_mm) <= tolerance_mm:
                return p  # éxito: posición real confirmada

            logger.warning(
                "ESP301 eje %d: WS terminó pero pos=%.6f, target=%.6f "
                "(intento %d/%d)",
                axis,
                p,
                position_mm,
                attempt,
                max_attempts,
            )

            # Jiggle antes de reintentar
            if attempt < max_attempts:
                jiggle = 0.001 if position_mm > 0 else -0.001
                self._send(f"{axis}PR{jiggle:.6f}")
                self._send_wait(f"{axis}WS", timeout_s=5.0)

        raise RuntimeError(
            f"ESP301 eje {axis}: falló después de {max_attempts} intentos. "
            f"Target: {position_mm:.6f} mm"
        )

    def stop_motion(self, axis: int) -> None:
        """Detener el movimiento sin bloquear esperando respuesta serial."""
        self._require_axis(axis)
        self._stop_requested.add(axis)
        # ST es una orden de emergencia. No usar _send(): ese método
        # intenta leer una respuesta y puede heredar el timeout largo de
        # _send_wait(), bloqueando el hilo de la GUI durante el movimiento.
        self._ser.write((f"{axis}ST\r").encode("ascii"))
        logger.warning("ESP301 eje %d: stop_motion solicitado", axis)

    def enable_axis(self, axis: int) -> None:
        self._require_axis(axis)
        self._send(f"{axis}MO")

    def disable_axis(self, axis: int) -> None:
        self._require_axis(axis)
        self._send(f"{axis}MF")

    def close(self) -> None:
        if self._ser:
            self._ser.close()
        self._ser = None

    @property
    def available_axes(self) -> set:
        return self._available_axes.copy()

    # ----------------------------------------------------------
    # Extra: velocidad (no está en la interfaz, específico ESP301)
    # ----------------------------------------------------------
    def set_velocity(self, axis: int, vel_mm_s: float) -> None:
        self._require_axis(axis)
        self._send(f"{axis}VA{vel_mm_s:.3f}")

    # ----------------------------------------------------------
    # Serial interno
    # ----------------------------------------------------------
    def _require_axis(self, axis: int) -> None:
        """Validar conexión y eje antes de ejecutar un comando físico."""
        if not self._ser or not self._ser.is_open:
            raise RuntimeError("ESP301: controlador no conectado")
        if axis not in AXIS_NAMES:
            raise ValueError(f"ESP301: eje inválido: {axis}")
        if self._available_axes and axis not in self._available_axes:
            raise RuntimeError(f"ESP301: eje {axis} no detectado")

    def _send(self, cmd: str, max_wait: float = 0.05) -> str:
        """Enviar comando ASCII + CR, espera adaptativa."""
        if not self._ser or not self._ser.is_open:
            return ""
        self._ser.write((cmd + "\r").encode("ascii"))

        start = time.time()
        while (time.time() - start) < max_wait:
            if self._ser.in_waiting > 0:
                return self._read()
            time.sleep(0.001)
        return self._read()

    def _send_wait(self, cmd: str, timeout_s: float = 30.0) -> None:
        """
        Enviar comando bloqueante (ej: WS) y esperar a que el
        controlador esté listo para recibir el siguiente comando.

        WS no devuelve respuesta; bloquea el flujo de comandos del
        ESP301 hasta que el motor frena. Para detectar que terminó,
        enviamos un comando inocuo (VE?) inmediatamente después:
        el controlador no lo procesa hasta que WS libera, y en ese
        momento responde con el string de versión.
        """
        if not self._ser or not self._ser.is_open:
            return
        # Enviar WS
        self._ser.write((cmd + "\r").encode("ascii"))
        # Enviar VE? (version query) como "señal" de que WS terminó
        self._ser.write(b"VE?\r")

        # Esperar la respuesta de VE? — llega solo cuando WS liberó
        old_timeout = self._ser.timeout
        self._ser.timeout = timeout_s
        try:
            resp = self._ser.readline().decode(errors="ignore").strip()
            if not resp:
                logger.warning(
                    "_send_wait: timeout después de %.1fs para '%s'",
                    timeout_s,
                    cmd,
                )
        finally:
            self._ser.timeout = old_timeout

    def _read(self) -> str:
        if not self._ser:
            return ""
        try:
            return self._ser.readline().decode(errors="ignore").strip()
        except Exception:
            return ""
