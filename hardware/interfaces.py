# -*- coding: utf-8 -*-
"""
hardware/interfaces.py
Interfaces abstractas de hardware.

Toda la capa de adquisición y la GUI hablan con estas
interfaces, nunca con las clases concretas directamente.

Contrato de unidades
--------------------
Las interfaces exponen unidades canónicas del software:

* posiciones mecánicas y tolerancias: milímetros (mm);
* tiempos: segundos (s), salvo métodos cuyo nombre indique milisegundos;
* el hardware puede usar otra unidad nativa (µm, pasos, cuentas, etc.),
  pero la conversión debe quedar encapsulada en su driver concreto.

El resto del software nunca debe convertir unidades nativas de un motor.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from constants import DEFAULT_POSITION_TOLERANCE_MM


class MotionCancelled(RuntimeError):
    """Movimiento detenido intencionalmente por el usuario."""


class SpectrometerInterface(ABC):

    @abstractmethod
    def capabilities(self) -> dict:
        """Capacidades estáticas declaradas por el driver."""
        ...

    @abstractmethod
    def connect(self) -> bool:
        """Conectar al espectrómetro. Retorna True si exitoso."""
        ...

    @abstractmethod
    def set_exposure_ms(self, value: float) -> None:
        """Configurar tiempo de integración en milisegundos."""
        ...

    def set_dark_correction(self, enabled: bool) -> None:
        """
        Activar/desactivar corrección de dark.
        Implementación por defecto: no-op (espectrómetros sin dark).
        """
        pass

    def set_nonlinearity_correction(self, enabled: bool) -> None:
        """
        Activar/desactivar corrección de no-linealidad del detector.
        Implementación por defecto: no-op (espectrómetros sin corrección).
        """
        pass

    @abstractmethod
    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Leer un espectro.

        Retorna
        -------
        (wavelengths_nm, intensities) : (ndarray, ndarray)

        Raises
        ------
        RuntimeError si el espectrómetro no responde o no está conectado.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Liberar recursos del espectrómetro."""
        ...

    def device_info(self) -> dict:
        """
        Información del dispositivo (modelo, serial, etc.).
        Default: {}. Implementaciones concretas pueden agregar
        campos según lo que el driver exponga.
        """
        return {}

    def reconnect(self) -> bool:
        """
        Cerrar handle actual y reconectar.
        Default: close() + connect(). Override si el driver
        necesita lógica especial (ej: handle stale en seabreeze).
        """
        self.close()
        return self.connect()


class MotionControllerInterface(ABC):
    """Controlador de movimiento con posiciones canónicas en mm.

    Cada implementación es responsable de convertir entre mm y la unidad
    nativa de su hardware antes de enviar o interpretar comandos.
    """

    @abstractmethod
    def capabilities(self) -> dict:
        """Capacidades estáticas declaradas por el driver."""
        ...

    @abstractmethod
    def connect(self) -> bool:
        """Conectar al controlador. Retorna True si exitoso."""
        ...

    @abstractmethod
    def move_absolute(self, axis: int, position_mm: float) -> None:
        """Mover en posición absoluta canónica (mm)."""
        ...

    @abstractmethod
    def get_position(self, axis: int) -> float:
        """
        Leer posición actual de un eje.

        Retorna
        -------
        float : posición en mm

        Raises
        ------
        RuntimeError si no se puede leer la posición.
        """
        ...

    @abstractmethod
    def goto_and_wait(
        self,
        axis: int,
        position_mm: float,
        tolerance_mm: float = DEFAULT_POSITION_TOLERANCE_MM,
        timeout_s: float = 30.0,
    ) -> float:
        """
        Mover a posición canónica y esperar hasta llegar.

        Parameters
        ----------
        position_mm : float
            Objetivo en milímetros, independientemente de la unidad nativa.
        tolerance_mm : float
            Tolerancia de llegada en milímetros.
        timeout_s : float
            Timeout de espera en segundos.

        Returns
        -------
        float : posición confirmada en milímetros después de terminar el
            movimiento y verificar la tolerancia.

        Raises
        ------
        RuntimeError (o MotorError) si falla el movimiento.
        """
        ...

    @abstractmethod
    def stop_motion(self, axis: int) -> None:
        """Detener el movimiento actual del eje sin deshabilitarlo."""
        ...

    @abstractmethod
    def enable_axis(self, axis: int) -> None: ...

    @abstractmethod
    def disable_axis(self, axis: int) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def available_axes(self) -> set:
        """Conjunto de ejes disponibles {1, 2, 3}."""
        ...
