# -*- coding: utf-8 -*-
"""
constants.py
Constantes y helpers compartidos del proyecto OCT.
"""

import math

MAX_WINDOWS = 5
SCHEMA_VERSION = "6.0.0"
SOFTWARE_NAME = "OCT_Static_Software"
SOFTWARE_VERSION = f"{SOFTWARE_NAME} V6.1"

# Defaults de selección de componentes y recorrido
DEFAULT_SPECTROMETER_KIND = "hr4000"
DEFAULT_MOTION_CONTROLLER_KIND = "esp301"
DEFAULT_SCAN_MODE = "snake"
DEFAULT_AXIS_ORDER = "XYZ"  # rápido → lento

# Defaults del controlador de motores
DEFAULT_MOTOR_PORT = "COM3"
DEFAULT_MOTOR_BAUD = 921600
DEFAULT_MOTOR_SERIAL_TIMEOUT_S = 0.1
DEFAULT_POSITION_TOLERANCE_UM = 2.0
MIN_POSITION_TOLERANCE_UM = 0.5
DEFAULT_POSITION_TOLERANCE_MM = DEFAULT_POSITION_TOLERANCE_UM * 1e-3

# Defaults compartidos del pipeline y del almacenamiento
DEFAULT_N_PIXELS = 3648
DEFAULT_K_SAMPLES = 3648
DEFAULT_PROFILE_SAMPLES = 2048
DEFAULT_Z_MAX_GLOBAL_M = 3.0e-3
DEFAULT_EXPOSURE_MS = 10.0
DEFAULT_SETTLING_TIME_S = 0.05

# Defaults exclusivos de simulación (no afectan hardware real)
MOCK_MOTION_SPEED_MM_S = 5.0
MOCK_DARK_COUNTS = 123.0

# Defaults ópticos de la GUI y metadata nueva
DEFAULT_FIBER_DIAMETER_UM = 5.0
DEFAULT_WAVELENGTH_NM = 850.0
DEFAULT_COLLIMATOR_FOCAL_LENGTH_MM = 9.0
DEFAULT_OBJECTIVE_FOCAL_LENGTH_MM = 18.0

# Defaults de guardado
DEFAULT_SAVE_FORMAT = "npz"
DEFAULT_SAVE_DIRECTORY = "Barridos Guardados"
DEFAULT_SAVE_SPECTRA = True
DEFAULT_SAVE_PEAKS = True
DEFAULT_SAVE_PROFILE = False
DEFAULT_PROFILE_MODE = "modulus"

# Mapeo entre nombres de eje y números de motor
AXIS_NUMBERS = {"X": 1, "Y": 2, "Z": 3}
AXIS_NAMES = {1: "X", 2: "Y", 3: "Z"}


def frange(a: float, b: float, s: float):
    """
    Generador de rango flotante inclusivo.

    Soporta dirección directa (a < b) e inversa (a > b).
    El step se toma en valor absoluto internamente.

    Yields
    ------
    float : valores desde a hasta b (inclusive) con paso s.
    """
    s = abs(s)
    if s == 0:
        yield a
        return

    n = count_range(a, b, s)
    direction = 1.0 if b >= a else -1.0
    tolerance = 1e-12 * max(1.0, abs(a), abs(b))

    # Generar desde un índice entero evita que el error acumulado de
    # repetir `v += s` haga divergir esta función de count_range().
    for i in range(n):
        value = a + direction * i * s
        # Si el último valor cae dentro de la tolerancia del extremo,
        # devolver exactamente el extremo evita residuos como
        # 0.30000000000000004.
        if abs(value - b) <= tolerance:
            value = b
        yield value


def count_range(a: float, b: float, s: float) -> int:
    """Cuenta puntos en un rango sin generarlos (O(1))."""
    s = abs(s)
    if s == 0:
        return 1
    quotient = abs((b - a) / s)
    # La tolerancia sólo corrige errores de representación flotante
    # alrededor de un cociente entero. No incluye el extremo cuando el
    # siguiente paso realmente lo supera.
    tolerance = 1e-12 * max(1.0, quotient)
    return math.floor(quotient + tolerance) + 1
