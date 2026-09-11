# -*- coding: utf-8 -*-
"""
gui/workers.py
QThreads para operaciones fuera del main thread.

Contiene:
  - SystemState: estados de la máquina de estados de la GUI
  - MonitorWorker: adquisición continua para modo monitor
  - ScanWorker: barrido (motores + adquisición + procesamiento)
  - MoveWorker: movimiento manual de motores
"""

import logging
import time
from enum import Enum, auto

from PyQt5.QtCore import QThread, pyqtSignal

from constants import AXIS_NUMBERS
from hardware.interfaces import MotionCancelled

logger = logging.getLogger(__name__)


# ============================================================
# Estados del sistema
# ============================================================
class SystemState(Enum):
    IDLE       = auto()
    MONITORING = auto()
    SCANNING   = auto()
    ERROR      = auto()


# ============================================================
# MonitorWorker
# ============================================================
class MonitorWorker(QThread):
    """
    Thread dedicado para modo monitor.
    Adquiere un espectro, procesa y emite el resultado.
    La GUI solo grafica (nunca procesa).
    """
    result_ready = pyqtSignal(object, object)  # (RawPointData, ProcessingResult)
    error_occurred = pyqtSignal(str)

    def __init__(self, acq_engine, proc_engine):
        super().__init__()
        self.acq = acq_engine
        self.proc = proc_engine
        self._running = False
        self._stop_requested = False
        self._max_consecutive_errors = 10

    def run(self):
        self._running = True
        consecutive_errors = 0
        while self._running and not self._stop_requested:
            try:
                raw = self.acq.acquire_monitor()
                result = self.proc.process(raw)
                self.result_ready.emit(raw, result)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                self.error_occurred.emit(str(e))
                if consecutive_errors >= self._max_consecutive_errors:
                    self.error_occurred.emit(
                        f"Monitor detenido: {consecutive_errors} errores "
                        f"consecutivos. Verificar hardware.")
                    self._running = False
                    break
            self.msleep(50)  # ~20 fps máximo

    def stop(self):
        self._stop_requested = True
        self._running = False


# ============================================================
# ScanWorker
# ============================================================
class ScanWorker(QThread):
    """
    Thread dedicado para barrido.
    Mueve motores, adquiere, procesa y emite resultado por punto.
    La GUI solo grafica y guarda (nunca adquiere ni procesa).
    """
    point_ready = pyqtSignal(float, float, float, object, object)
    # (x_mm, y_mm, z_mechanical_mm, RawPointData, ProcessingResult)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    aborted = pyqtSignal()

    def __init__(
        self,
        mot,
        acq_engine,
        proc_engine,
        scan_config,
        measurements_per_point,
        position_tolerance_mm,
    ):
        super().__init__()
        self.mot = mot
        self.acq = acq_engine
        self.proc = proc_engine
        self.scan_config = scan_config
        self.measurements_per_point = measurements_per_point
        self.position_tolerance_mm = position_tolerance_mm
        self._abort_requested = False

    def abort(self):
        self._abort_requested = True
        logger.warning("Abort solicitado — terminando punto actual")

    def run(self):
        cfg = self.scan_config
        error = None
        try:
            # Leer posición real de ejes inactivos (una sola vez).
            # scan_cfg no se modifica: las posiciones se pasan como
            # parámetro externo a iter_points.
            inactive_pos = {}
            for axis_name in ("X", "Y", "Z"):
                use, _, _, _ = cfg._axis_params(axis_name)
                if not use:
                    try:
                        motor_num = AXIS_NUMBERS[axis_name]
                        inactive_pos[axis_name] = self.mot.get_position(motor_num)
                    except Exception as exc:
                        raise RuntimeError(
                            f"No se pudo leer la posición del eje inactivo {axis_name}: {exc}"
                        ) from exc

            prev = {"X": None, "Y": None, "Z": None}
            actual_positions = inactive_pos.copy()

            # Orden de movimiento: lento primero, rápido último.
            # Se mueve solo si el eje cambió respecto al punto anterior.
            move_order = list(reversed(cfg.active_axes))

            for x_mm, y_mm, z_mechanical_mm in cfg.iter_points(inactive_pos=inactive_pos):
                if self._abort_requested:
                    break

                current = {
                    "X": x_mm,
                    "Y": y_mm,
                    "Z": z_mechanical_mm,
                }

                # Mover en orden lento → rápido, solo ejes que cambiaron
                for axis_name in move_order:
                    val = current[axis_name]
                    if val != prev[axis_name]:
                        motor_num = AXIS_NUMBERS[axis_name]
                        self.mot.goto_and_wait(
                            motor_num,
                            val,
                            tolerance_mm=self.position_tolerance_mm,
                        )
                        time.sleep(cfg.settling_time_s)
                        # La coordenada que acompaña la medición debe ser la
                        # lectura posterior al settling mecánico.
                        actual_positions[axis_name] = self.mot.get_position(motor_num)

                prev = current.copy()

                x_actual = actual_positions["X"]
                y_actual = actual_positions["Y"]
                z_actual = actual_positions["Z"]

                # Adquisición + procesamiento en ESTE thread (no main)
                raw = self.acq.acquire_point(
                    x_actual,
                    y_actual,
                    z_actual,
                    self.measurements_per_point,
                )
                result = self.proc.process(raw)
                self.point_ready.emit(
                    x_actual,
                    y_actual,
                    z_actual,
                    raw,
                    result,
                )

        except Exception as e:
            error = e

        # Retorno al origen: se intenta SIEMPRE (error, abort o
        # finalización normal). Si el retorno falla, se logea pero
        # no enmascara el error original.
        try:
            logger.info("Retornando motores al origen")
            for axis_name in reversed(cfg.active_axes):
                _, start, _, _ = cfg._axis_params(axis_name)
                motor_num = AXIS_NUMBERS[axis_name]
                self.mot.goto_and_wait(
                    motor_num,
                    start,
                    tolerance_mm=self.position_tolerance_mm,
                )
            logger.info("Motores en posición inicial")
        except Exception as e_ret:
            logger.error("Retorno al origen falló: %s", e_ret, exc_info=True)
            if error is None:
                error = RuntimeError(f"Retorno al origen falló: {e_ret}")

        # Señales: una sola, después del retorno.
        if error is not None:
            self.error_occurred.emit(str(error))
        elif self._abort_requested:
            logger.warning("Barrido ABORTADO por usuario")
            self.aborted.emit()
        else:
            self.finished.emit()


# ============================================================
# MoveWorker
# ============================================================
class MoveWorker(QThread):
    """
    Thread dedicado para movimiento manual de motores.
    Ejecuta goto_and_wait fuera del main thread para no congelar la GUI.

    Soporta uno o varios ejes (lista de (axis, target_pos)), de modo que
    también sirve para "Home (0,0,0)" sin bloquear la GUI.
    """
    finished_ok = pyqtSignal(list)        # [(axis:int, actual:float), ...]
    stopped = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, mot, moves, position_tolerance_mm):
        """
        Parámetros
        ----------
        mot   : controlador de motores (con goto_and_wait y get_position).
        moves : lista de tuplas (axis:int, target:float).
        """
        super().__init__()
        self.mot = mot
        self.moves = list(moves)
        self.position_tolerance_mm = position_tolerance_mm
        self._stop_requested = False

    def stop(self):
        """Solicitar parada inmediata del movimiento manual actual."""
        self._stop_requested = True
        for axis, _target in self.moves:
            try:
                self.mot.stop_motion(axis)
            except Exception as exc:
                logger.error("No se pudo detener el eje %d: %s", axis, exc)

    def run(self):
        results = []
        try:
            for axis, target in self.moves:
                if self._stop_requested:
                    self.stopped.emit()
                    return
                self.mot.goto_and_wait(
                    axis,
                    target,
                    tolerance_mm=self.position_tolerance_mm,
                )
                actual = self.mot.get_position(axis)
                results.append((axis, actual))
            self.finished_ok.emit(results)
        except MotionCancelled:
            logger.warning("Movimiento manual detenido por usuario")
            self.stopped.emit()
        except Exception as e:
            logger.error("MoveWorker error: %s", e, exc_info=True)
            self.error_occurred.emit(str(e))
