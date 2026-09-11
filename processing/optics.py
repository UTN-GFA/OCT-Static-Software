# -*- coding: utf-8 -*-
"""
processing/optics.py
Cálculos ópticos del sistema OCT.

Funciones puras para propagar una cintura gaussiana desde la fibra
hasta el foco del objetivo y calcular el parámetro confocal. No
depende de ningún otro módulo del pipeline.

Cadena óptica modelada:
    Fibra (fiber_diameter_um) → Colimador (collimator_focal_length_mm)
    → Haz colimado → Objetivo (objective_focal_length_mm) → Muestra

Parámetros de entrada:
    fiber_diameter_um  : diámetro del núcleo de la fibra (µm)
    wavelength_nm       : longitud de onda central (nm)
    collimator_focal_length_mm : distancia focal del colimador (mm)
    objective_focal_length_mm  : distancia focal del objetivo (mm)
"""

from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OpticsResult:
    """
    Resultado del cálculo óptico del sistema.
    Todos los valores en las unidades indicadas.
    """

    # Inputs (preservados para reproducibilidad)
    fiber_diameter_um: float
    wavelength_nm: float
    collimator_focal_length_mm: float
    objective_focal_length_mm: float

    # Outputs
    beam_diameter_mm: float  # diámetro del haz colimado
    na_effective: float  # NA efectivo en la muestra
    spot_gaussian_um: float  # diámetro de cintura gaussiana (2·Ws)
    confocal_parameter_um: float  # parámetro confocal


def calculate_optics(
    fiber_diameter_um: float,
    wavelength_nm: float,
    collimator_focal_length_mm: float,
    objective_focal_length_mm: float,
) -> OpticsResult:
    """
    Calcular parámetros ópticos del sistema OCT.

    Parámetros
    ----------
    fiber_diameter_um : float
        Diámetro del núcleo de la fibra (µm).
        Para fibras monomodo, idealmente usar MFD.
        Valor típico: 5.0 µm (SM600, ~850 nm).
    wavelength_nm : float
        Longitud de onda central (nm). Típico: 850 nm.
    collimator_focal_length_mm : float
        Distancia focal del colimador (mm).
    objective_focal_length_mm : float
        Distancia focal del objetivo (mm).

    Retorna
    -------
    OpticsResult con todos los parámetros calculados.

    Raises
    ------
    ValueError si algún parámetro es <= 0.
    """
    # ── Validación ────────────────────────────────────────────
    if fiber_diameter_um <= 0:
        raise ValueError(f"fiber_diameter_um debe ser > 0 (recibió {fiber_diameter_um})")
    if wavelength_nm <= 0:
        raise ValueError(f"wavelength_nm debe ser > 0 (recibió {wavelength_nm})")
    if collimator_focal_length_mm <= 0:
        raise ValueError(f"collimator_focal_length_mm debe ser > 0 (recibió {collimator_focal_length_mm})")
    if objective_focal_length_mm <= 0:
        raise ValueError(f"objective_focal_length_mm debe ser > 0 (recibió {objective_focal_length_mm})")

    # ── Propagación gaussiana hasta el colimador ───────────────
    # W0 es el radio de cintura en la fibra; el dato de la GUI es
    # el diámetro. La formulación sigue el cálculo W0y_V2.py:
    #   Wz = W0 · sqrt(1 + (λ·f_col/(π·W0²))²)
    fiber_waist_m = fiber_diameter_um * 1e-6 / 2.0
    collimator_focal_m = collimator_focal_length_mm * 1e-3
    objective_focal_m = objective_focal_length_mm * 1e-3
    wavelength_m = wavelength_nm * 1e-9
    wz_m = fiber_waist_m * math.sqrt(
        1.0
        + (
            wavelength_m
            * collimator_focal_m
            / (math.pi * fiber_waist_m**2)
        )
        ** 2
    )
    beam_diameter_mm = 2.0 * wz_m * 1e3

    # ── NA efectivo ───────────────────────────────────────────
    # NA = D_beam / (2 · f_obj)
    na_effective = beam_diameter_mm / (2.0 * objective_focal_length_mm)

    # ── Cintura gaussiana en el foco del objetivo ──────────────
    # Resolver en y = Ws²:
    #   y² - Wz²·y + (λ²·f_obj²/π²) = 0
    # Se toma la raíz menor positiva, que corresponde a la cintura
    # focal. La raíz grande representa la solución no enfocada.
    constant = (wavelength_m * objective_focal_m / math.pi) ** 2
    discriminant = wz_m**4 - 4.0 * constant
    tolerance = max(wz_m**4, 1.0) * 1e-15
    if discriminant < -tolerance:
        raise ValueError(
            "Los parámetros no producen una cintura gaussiana real "
            "en el objetivo."
        )
    discriminant = max(0.0, discriminant)
    waist_squared_m2 = (wz_m**2 - math.sqrt(discriminant)) / 2.0
    if waist_squared_m2 <= 0.0:
        raise ValueError("La cintura gaussiana calculada no es positiva.")
    sample_waist_m = math.sqrt(waist_squared_m2)
    spot_gaussian_um = 2.0 * sample_waist_m * 1e6

    # ── Parámetro confocal ─────────────────────────────────────
    confocal_parameter_um = (
        2.0 * math.pi * sample_waist_m**2 / wavelength_m * 1e6
    )

    return OpticsResult(
        fiber_diameter_um=fiber_diameter_um,
        wavelength_nm=wavelength_nm,
        collimator_focal_length_mm=collimator_focal_length_mm,
        objective_focal_length_mm=objective_focal_length_mm,
        beam_diameter_mm=beam_diameter_mm,
        na_effective=na_effective,
        spot_gaussian_um=spot_gaussian_um,
        confocal_parameter_um=confocal_parameter_um,
    )
