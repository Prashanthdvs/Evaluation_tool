@echo off
setlocal EnableDelayedExpansion
set ROOT=%~dp0
set VENV=%ROOT%.venv\Scripts\python.exe
set PORT=8502

echo.
echo  ============================================================
echo   Decision Engine -- MT Provider Selection Engine
echo   Deploying on Windows VM
echo  ============================================================
echo.

:: ── Check .venv exists ───────────────────────────────────────
if not exist "%VENV%" (
    echo  [ERROR] Virtual environment not found at %ROOT%.venv
    echo  Run setup.bat first to create the environment.
    pause
    exit /b 1
)

:: ── Kill any process already on PORT ────────────────────────
echo  [1/3] Checking port %PORT%...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT% " ^| findstr "LISTENING" 2^>nul') do (
    echo         Killing existing process on port %PORT% (PID %%p)
    taskkill /PID %%p /F >nul 2>&1
)

:: ── Kill any lingering python/streamlit processes ────────────
taskkill /IM "python.exe" /F >nul 2>&1

timeout /t 1 /nobreak >nul

:: ── Start Streamlit (FastAPI backend is embedded inside) ─────
echo  [2/3] Starting Decision Engine on port %PORT%...
echo.

start "Decision-Engine" cmd /k "cd /d %ROOT% && echo Starting... && %VENV% -m streamlit run streamlit_app.py --server.port %PORT% --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false"

:: ── Wait for app to bind ─────────────────────────────────────
echo  [3/3] Waiting for application to start...
timeout /t 6 /nobreak >nul

echo.
echo  ============================================================
echo   Application is LIVE
echo.
echo   Local  : http://localhost:%PORT%
echo   Network: http://%COMPUTERNAME%:%PORT%
echo   API    : http://localhost:8000/docs
echo  ============================================================
echo.

:: Open browser
start http://localhost:%PORT%

echo  Press any key to exit this window (app keeps running).
pause >nul
echo   Backend : http://localhost:8000
echo   Frontend: http://localhost:5173
echo ====================================================
echo.
timeout /t 3
start http://localhost:5173
