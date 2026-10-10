@echo off
rem Windows launcher: first run creates .venv and installs dependencies.
rem ASCII only on purpose: non-ASCII text breaks cmd parsing on Chinese-locale Windows.
cd /d "%~dp0"
where ffmpeg >nul 2>nul || (echo [ERROR] ffmpeg not found. Run: winget install Gyan.FFmpeg  then open a new window. & pause & exit /b 1)
if not exist .venv\Scripts\python.exe (
  echo First run: creating Python 3.14 environment...
  py -3.14 -m venv .venv 2>nul || python -m venv .venv
)
if not exist .venv\Scripts\python.exe (echo [ERROR] Python not found. Run: py install 3.14 & pause & exit /b 1)
echo Installing dependencies, first run takes 1-3 minutes...
.venv\Scripts\python -m pip install -q -r requirements.txt || (echo [ERROR] Dependency install failed. Screenshot this window and send it to Claude. & pause & exit /b 1)
.venv\Scripts\python app.py
pause
