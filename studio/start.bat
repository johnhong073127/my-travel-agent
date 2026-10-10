@echo off
rem Windows 一键启动：首次运行自动建虚拟环境并安装依赖
chcp 65001 >nul
cd /d "%~dp0"
where ffmpeg >nul 2>nul || (echo 缺少 ffmpeg：请运行 winget install Gyan.FFmpeg 后重新打开本窗口 & pause & exit /b 1)
if not exist .venv\Scripts\python.exe (
  rem 优先用 Python 3.14：最新版刚发布时，numpy / OpenCV 往往还没有对应的安装包
  py -3.14 -m venv .venv 2>nul || python -m venv .venv
)
if not exist .venv\Scripts\python.exe (echo 找不到 Python：请先运行 py install 3.14 & pause & exit /b 1)
.venv\Scripts\python -m pip install -q -r requirements.txt || (echo 依赖安装失败，请把上面的报错截图发给 Claude & pause & exit /b 1)
.venv\Scripts\python app.py
pause
