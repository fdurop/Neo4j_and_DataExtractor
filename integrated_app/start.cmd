@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup.ps1 first to install the Python environment.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -X utf8 serve.py
if errorlevel 1 pause
