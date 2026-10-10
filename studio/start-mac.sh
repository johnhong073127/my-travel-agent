#!/usr/bin/env bash
# macOS / Linux 一键启动：首次运行自动建虚拟环境并安装依赖
set -e
cd "$(dirname "$0")"
command -v ffmpeg >/dev/null || { echo "缺少 ffmpeg：macOS 运行 brew install ffmpeg；Ubuntu 运行 sudo apt install ffmpeg"; exit 1; }
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
exec .venv/bin/python app.py "$@"
