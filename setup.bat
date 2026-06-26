@echo off
setlocal
set ROOT=%~dp0

echo ====================================================
echo   MT Evaluation Engine - First-time Setup
echo ====================================================

:: ── Python venv ──────────────────────────────────────
echo.
echo [1/3] Creating Python virtual environment...
cd /d %ROOT%backend
if not exist .venv (
    python -m venv .venv
    echo     venv created.
) else (
    echo     venv already exists, skipping.
)

echo [2/3] Installing Python dependencies...
call .venv\Scripts\activate.bat
pip install -r requirements.txt
echo     Python packages installed.

:: ── Frontend npm ─────────────────────────────────────
echo.
echo [3/3] Installing Node.js dependencies...
cd /d %ROOT%frontend
npm install
echo     Node packages installed.

echo.
echo ====================================================
echo   Setup complete!
echo   Run start.bat to launch the application.
echo ====================================================
pause
