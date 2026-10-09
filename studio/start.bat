@echo off
rem Windows 一键启动：首次运行自动建虚拟环境并安装依赖
chcp 65001 >nul
cd /d "%~dp0"
where ffmpeg >nul 2>nul || (echo 缺少 ffmpeg：请运行 winget install Gyan.FFmpeg 后重新打开本窗口 & pause & exit /b 1)
if not exist .venv python -m venv .venv
.venv\Scripts\pip install -q -r requirements.txt
.venv\Scripts\python app.py
pause
