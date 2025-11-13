@echo off
echo Starting YoopRL Backend API...
echo.
REM Change to script directory (project root) regardless of current drive
cd /d "%~dp0"

REM Ensure Python can import the project packages
set PYTHONPATH=%~dp0

REM Prefer the project's venv Python if available, otherwise try py -3.11
if exist ".venv\Scripts\python.exe" (
	.venv\Scripts\python.exe backend\api\main.py
) else (
	py -3.11 backend\api\main.py
)

pause
