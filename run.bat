@echo off
title DermVision AI - Skin Disease Detection Platform
color 0B
echo =====================================================================
echo                 DERMVISION AI STARTUP CONTROL
echo =====================================================================
echo.
echo [1/3] Checking environment dependencies...
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please make sure the 'venv' folder exists in the project root.
    echo.
    pause
    exit /b
)

echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/3] Checking PyTorch model weights...
if not exist "skin_disease_model.pth" (
    echo [INFO] skin_disease_model.pth not found!
    echo Generating default neural network weights using train.py...
    python train.py
) else (
    echo [OK] Located skin_disease_model.pth.
)

echo.
echo =====================================================================
echo               DermVision AI Server is launching...
echo =====================================================================
echo.
echo  * Server URL: http://127.0.0.1:5000/
echo  * Preprocessing: Dull Razor + CLAHE active
echo  * Deep Learning Backbone: MobileNetV2 + Squeeze-and-Excitation active
echo  * Scan History: sqlite3 (scans.db) active
echo.
echo Press Ctrl+C in this terminal to shut down the server.
echo =====================================================================
echo.

python app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Server crashed or stopped with exit code %errorlevel%.
    pause
)
