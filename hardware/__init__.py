# -*- coding: utf-8 -*-
"""
hardware/__init__.py
Registry y factory de hardware.

Para agregar un nuevo dispositivo:
  1. Crear hardware/spectrometer_xxx.py o hardware/motor_xxx.py
  2. Implementar SpectrometerInterface o MotionControllerInterface
  3. Agregar una entrada al registry correspondiente acá abajo

Los drivers físicos se importan de forma diferida. Cada driver debe exponer
el contrato canónico de interfaces.py; las conversiones desde la unidad nativa
del hardware se realizan exclusivamente dentro de su propio módulo.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from .spectrometer_mock import MockSpectrometer
from .motor_mock import MockMotionController
from constants import DEFAULT_MOTION_CONTROLLER_KIND, DEFAULT_SPECTROMETER_KIND


@dataclass(frozen=True)
class HardwareRegistration:
    """Metadata mínima para que la GUI descubra un driver registrado."""

    label: str
    entry: Any


# ── Registries ────────────────────────────────────────────────
# La entrada `entry` puede ser una clase ya cargada (mock) o una referencia
# (módulo, clase) para conservar imports diferidos de hardware físico.
#
# Para agregar hardware nuevo sólo hace falta crear el driver e incorporar
# una HardwareRegistration acá. La GUI y el procesamiento no se modifican.
SPECTROMETERS = {
    "hr4000": HardwareRegistration(
        "Ocean Optics HR4000",
        ("hardware.spectrometer_hr4000", "HR4000Spectrometer"),
    ),
    "mock": HardwareRegistration("Espectrómetro simulado", MockSpectrometer),
}

MOTION_CONTROLLERS = {
    "esp301": HardwareRegistration(
        "Newport ESP301",
        ("hardware.motor_esp301", "ESP301MotionController"),
    ),
    "mock": HardwareRegistration(
        "Controlador de motores simulado", MockMotionController
    ),
}


def _resolve_entry(entry: Any):
    if isinstance(entry, HardwareRegistration):
        entry = entry.entry
    if isinstance(entry, tuple):
        module_name, class_name = entry
        module = import_module(module_name)
        return getattr(module, class_name)
    return entry


def _list_registered_devices(registry: dict) -> list[tuple[str, str]]:
    return [
        (kind, registration.label)
        for kind, registration in registry.items()
    ]


def list_spectrometers() -> list[tuple[str, str]]:
    """Retornar (kind, etiqueta) sin importar drivers físicos."""
    return _list_registered_devices(SPECTROMETERS)


def list_motion_controllers() -> list[tuple[str, str]]:
    """Retornar (kind, etiqueta) sin importar drivers físicos."""
    return _list_registered_devices(MOTION_CONTROLLERS)


# ── Factories ─────────────────────────────────────────────────
def create_spectrometer(kind: str = DEFAULT_SPECTROMETER_KIND, **kwargs):
    """Crear espectrómetro por registry. Kwargs se pasan al constructor."""
    registration = SPECTROMETERS.get(kind)
    if registration is None:
        available = ", ".join(SPECTROMETERS.keys())
        raise ValueError(
            f"Espectrómetro '{kind}' no registrado. Disponibles: {available}"
        )
    return _resolve_entry(registration)(**kwargs)


def create_motion_controller(kind: str = DEFAULT_MOTION_CONTROLLER_KIND, **kwargs):
    """Crear controlador por registry. Kwargs se pasan al constructor."""
    registration = MOTION_CONTROLLERS.get(kind)
    if registration is None:
        available = ", ".join(MOTION_CONTROLLERS.keys())
        raise ValueError(
            f"Controlador '{kind}' no registrado. Disponibles: {available}"
        )
    return _resolve_entry(registration)(**kwargs)
