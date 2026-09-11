# Extender hardware sin tocar la GUI

La GUI descubre los dispositivos desde los registros de `hardware/__init__.py`.
Un driver nuevo no necesita agregar widgets ni modificar el procesamiento OCT.

## Agregar un espectrómetro

1. Crear `hardware/spectrometer_3.py`.
2. Implementar `SpectrometerInterface`.
3. Implementar `capabilities()` y declarar solamente las funciones realmente soportadas.
4. Mantener el contrato canónico:
   - `read()` devuelve `(wavelengths_nm, intensities)`;
   - la exposición se recibe en milisegundos;
   - las conversiones propias del SDK quedan dentro del driver.
5. Registrar el driver en `hardware/__init__.py`:

```python
SPECTROMETERS["spectrometer_3"] = HardwareRegistration(
    "Fabricante Modelo 3",
    ("hardware.spectrometer_3", "Spectrometer3"),
)
```

El import del SDK debe quedar dentro de `hardware/spectrometer_3.py` y cargarse
sólo cuando la factory resuelva ese registro.

Ejemplo mínimo de capacidades:

```python
def capabilities(self) -> dict:
    return {
        "exposure_control": True,
        "dark_correction": False,
        "nonlinearity_correction": True,
        "wavelengths_nm": True,
        "reconnect": True,
    }
```

## Agregar un controlador de motores

1. Crear `hardware/motor_4.py`.
2. Implementar `MotionControllerInterface`.
3. Implementar `capabilities()`.
4. Convertir dentro del driver las unidades nativas a las unidades canónicas del software:
   - posiciones y tolerancias en mm;
   - timeout en segundos.
5. Registrar el driver:

```python
MOTION_CONTROLLERS["motor_4"] = HardwareRegistration(
    "Fabricante Controlador 4",
    ("hardware.motor_4", "Motor4Controller"),
)
```

Ejemplo mínimo de capacidades:

```python
def capabilities(self) -> dict:
    return {
        "absolute_move": True,
        "relative_move": False,
        "position_readback": True,
        "stop_motion": True,
        "axis_enable": True,
        "homing": True,
        "velocity_control": False,
        "limits": True,
    }
```

El constructor debe aceptar `port=...` si el controlador usa puerto serie,
o adaptar ese parámetro en su propia configuración. La GUI siempre pasa el
puerto seleccionado a la factory; no conoce la clase concreta.

## Qué ocurre automáticamente

Al registrar la entrada:

- el dispositivo aparece en el selector de la pestaña **Drivers**;
- al seleccionarlo, la GUI cierra el driver anterior y crea el nuevo;
- se conecta usando la factory;
- se reinician las cachés dependientes del espectrómetro;
- los controles se habilitan o deshabilitan según `capabilities()`;
- los ejes se actualizan desde `available_axes` después de conectar.

No agregar condiciones como `if kind == "motor_4"` en `gui/main_gui.py`.
Si el driver necesita parámetros adicionales, deben resolverse en el driver o
en una configuración común, no mediante una rama específica de la GUI.
