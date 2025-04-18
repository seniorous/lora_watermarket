@echo off
chcp 65001 > nul

echo [INFO] Starting Watermark Tool...

:: Set virtual environment folder name
SET VENV_DIR=venv

:: Create new virtual environment if it doesn't exist
IF NOT EXIST %VENV_DIR% (
    echo [INFO] Creating virtual environment...
    python -m venv %VENV_DIR%
)

:: Activate virtual environment
CALL %VENV_DIR%\Scripts\activate

:: Install dependencies
echo [INFO] Installing requirements...
:: Use the recommended way to upgrade pip if needed
:: python -m pip install --upgrade pip
pip install -r requirements.txt

:: Launch application
echo [INFO] Launching application...
python app.py

IF errorlevel 1 (
    echo [ERROR] Application failed to start
    pause
    exit /b 1
)

pause
