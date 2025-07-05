@echo off
title AI-Based Accident Prevention System
color 0A

echo.
echo ================================================================
echo    AI-Based Accident Prevention System
echo ================================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://python.org
    pause
    exit /b 1
)

:: Check if we're in the right directory
if not exist "app.py" (
    echo ❌ app.py not found in current directory
    echo Please run this script from the project directory
    pause
    exit /b 1
)

:: Install dependencies if needed
echo 📦 Checking and installing dependencies...
pip install -r requirements.txt >nul 2>&1

:: Run the startup script
echo 🚀 Starting the application...
python run_app.py

:: If the startup script fails, try direct streamlit run
if errorlevel 1 (
    echo.
    echo ⚠️  Startup script failed, trying direct launch...
    echo.
    streamlit run app.py --server.port 8501 --server.address localhost
)

echo.
echo 👋 Application closed.
pause 