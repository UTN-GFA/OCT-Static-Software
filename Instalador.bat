@echo off
cd /d "%~dp0"

title Instalador OCT_Static_Software V6.1

echo ==========================================
echo OCT_Static_Software V6.1
echo ==========================================
echo.

REM =====================================================
REM Buscar Python 3.12
REM =====================================================

py -3.12 --version >nul 2>&1

if errorlevel 1 (
    echo Python 3.12 no encontrado.
    echo.
    echo Instalando Python 3.12.10...
    echo.

    start /wait installers\python-3.12.10-amd64.exe /passive InstallAllUsers=1 PrependPath=1

    echo.
    echo Reintentando deteccion...
    timeout /t 5 >nul

    py -3.12 --version >nul 2>&1

    if errorlevel 1 (
        echo.
        echo ERROR:
        echo No se pudo detectar Python 3.12.
        pause
        exit /b 1
    )
)

echo Python 3.12 detectado.
echo.

REM =====================================================
REM Crear entorno virtual
REM =====================================================

if not exist ".venv" (
    echo Creando entorno virtual...
    py -3.12 -m venv ".venv"

    if errorlevel 1 (
        echo.
        echo ERROR creando entorno virtual.
        pause
        exit /b 1
    )
) else (
    echo Entorno virtual existente encontrado.
)

echo.

REM =====================================================
REM Activar entorno virtual
REM =====================================================

call ".venv\Scripts\activate.bat"

if errorlevel 1 (
    echo.
    echo ERROR activando entorno virtual.
    pause
    exit /b 1
)

echo.
echo Configurando pip para redes institucionales...

set PIP_TRUST=--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org

echo.
echo Actualizando pip...
python -m pip install --upgrade pip %PIP_TRUST%

echo.
echo Instalando dependencias...
python -m pip install -r installers\requirements.txt %PIP_TRUST%

if errorlevel 1 (
    echo.
    echo ERROR instalando dependencias.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Instalacion completada correctamente
echo ==========================================
echo.
pause
