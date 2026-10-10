"""影片工作室：本地网页应用。python3 app.py 后浏览器打开 http://127.0.0.1:8765"""
import asyncio
import json
import mimetypes
import os
import re
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2

import engine

ROOT = Path(__file__).resolve().parent
PROJ = ROOT / 'projects' / 'main'
for d in ('assets', 'cache', 'out'):
    (PROJ / d).mkdir(parents=True, exist_ok=True)
PROJECT_FILE = PROJ / 'project.json'

VOICES = [
    {'id': 'zh-CN-YunjianNeural', 'name': '云健 · 男 · 浑厚有力（推荐史诗旁白）'},
    {'id': 'zh-CN-YunyangNeural', 'name': '云扬 · 男 · 新闻播音'},
    {'id': 'zh-CN-YunxiNeural', 'name': '云希 · 男 · 青年'},
    {'id': 'zh-CN-YunxiaNeural', 'name': '云夏 · 男 · 少年'},
    {'id': 'zh-CN-XiaoxiaoNeural', 'name': '晓晓 · 女 · 温暖'},
    {'id': 'zh-CN-XiaoyiNeural', 'name': '晓伊 · 女 · 活泼'},
    {'id': 'zh-TW-YunJheNeural', 'name': '雲哲 · 男 · 台湾腔'},
    {'id': 'zh-HK-WanLungNeural', 'name': '雲龍 · 男 · 粤语'},
]

job = {'state': 'idle', 'message': '', 'progress': 0.0, 'output': None, 'error': None}
lock = threading.Lock()


def load_project():
    if PROJECT_FILE.exists():
        return json.loads(PROJECT_FILE.read_text('utf-8'))
    return {'title': '未命名影片', 'shots': [], 'narration': [], 'music': None}


def save_project(p):
    tmp = PROJECT_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(p, ensure_ascii=False, indent=2), 'utf-8')
    tmp.replace(PROJECT_FILE)


def safe_name(name):
    stem, ext = os.path.splitext(os.path.basename(name or 'file'))
    stem = re.sub(r'[^\w.-]', '_', stem)[:40]
    return f'{uuid.uuid4().hex[:8]}_{stem}{ext.lower()}'


def tts(text, voice, rate, pitch):
    try:
        import edge_tts
    except ImportError:
        raise RuntimeError('未安装 edge-tts：请运行 pip install edge-tts')
    name = f'assets/{uuid.uuid4().hex[:8]}_tts.mp3'
    comm = edge_tts.Communicate(text, voice, rate=f'{int(rate):+d}%', pitch=f'{int(pitch):+d}Hz',
                                proxy=os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy'))
    try:
        asyncio.run(comm.save(str(PROJ / name)))
    except Exception as e:
        raise RuntimeError(f'配音服务连接失败（需要能访问 speech.platform.bing.com）：{e}')
    return name


def run_render(project):
    def progress(msg, frac):
        with lock:
            job.update(message=msg, progress=round(frac, 4))
    try:
        title = re.sub(r'[^\w-]', '_', project.get('title') or 'film')
        out = f'out/{title}_{time.strftime("%m%d_%H%M%S")}.mp4'
        total = engine.render_film(project, str(PROJ), str(PROJ / out), progress)
        with lock:
            job.update(state='done', output=out, message=f'完成 · {total:.1f} 秒', progress=1.0)
    except Exception as e:
        traceback.print_exc()
        with lock:
            job.update(state='error', error=str(e), message='渲染失败')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass

    def send(self, code, body=b'', ctype='application/json; charset=utf-8', headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        return self.rfile.read(int(self.headers.get('Content-Length') or 0))

    def json(self):
        return json.loads(self.body() or b'{}')

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/':
            return self.send(200, (ROOT / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        if u.path == '/api/project':
            return self.send(200, load_project())
        if u.path == '/api/voices':
            return self.send(200, VOICES)
        if u.path == '/api/status':
            with lock:
                return self.send(200, dict(job))
        if u.path.startswith('/files/'):
            return self.serve_file(unquote(u.path[len('/files/'):]))
        self.send(404, {'error': 'not found'})

    def serve_file(self, rel):
        f = (PROJ / rel).resolve()
        if PROJ.resolve() not in f.parents or not f.is_file():
            return self.send(404, {'error': 'not found'})
        size = f.stat().st_size
        ctype = mimetypes.guess_type(f.name)[0] or 'application/octet-stream'
        m = re.match(r'bytes=(\d*)-(\d*)', self.headers.get('Range', ''))
        with open(f, 'rb') as fh:
            if m and (m.group(1) or m.group(2)):
                start = int(m.group(1)) if m.group(1) else size - int(m.group(2))
                end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
                end = min(end, size - 1)
                fh.seek(start)
                return self.send(206, fh.read(end - start + 1), ctype,
                                 {'Content-Range': f'bytes {start}-{end}/{size}', 'Accept-Ranges': 'bytes'})
            return self.send(200, fh.read(), ctype, {'Accept-Ranges': 'bytes'})

    def do_PUT(self):
        if urlparse(self.path).path == '/api/project':
            save_project(self.json())
            return self.send(200, {'ok': True})
        self.send(404, {'error': 'not found'})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            if u.path == '/api/upload':
                name = 'assets/' + safe_name(parse_qs(u.query).get('name', ['file'])[0])
                (PROJ / name).write_bytes(self.body())
                info = {'path': name}
                if not re.search(r'\.(png|jpe?g|webp|bmp)$', name):
                    info['duration'] = round(engine.audio_duration(str(PROJ / name)), 2)
                return self.send(200, info)
            if u.path == '/api/tts':
                q = self.json()
                if not q.get('text', '').strip():
                    return self.send(400, {'error': '旁白文字为空'})
                name = tts(q['text'], q.get('voice') or VOICES[0]['id'], q.get('rate', -20), q.get('pitch', -8))
                return self.send(200, {'path': name, 'duration': round(engine.audio_duration(str(PROJ / name)), 2)})
            if u.path == '/api/preview':
                sh = self.json()
                shot = engine.Shot(str(PROJ / sh['image']), sh, scale=0.3)
                n = int(round(shot.D * engine.FPS))
                hold = engine.shot_params(sh)['hold']
                tag = uuid.uuid4().hex[:6]
                urls = []
                for k, i in enumerate([0, n // 3, int(n * (1 - hold / shot.D)) - 1, n - 1]):
                    rel = f'cache/preview_{k}_{tag}.jpg'
                    # cv2.imwrite 不支持中文路径：先编码再写字节
                    (PROJ / rel).write_bytes(cv2.imencode('.jpg', shot.frame(max(0, i)), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
                    urls.append({'path': rel, 't': round(max(0, i) / engine.FPS, 2)})
                for old in (PROJ / 'cache').glob('preview_*.jpg'):
                    if tag not in old.name and time.time() - old.stat().st_mtime > 60:
                        old.unlink(missing_ok=True)
                return self.send(200, urls)
            if u.path == '/api/render':
                with lock:
                    if job['state'] == 'running':
                        return self.send(409, {'error': '已有渲染在进行'})
                    job.update(state='running', message='准备中', progress=0.0, output=None, error=None)
                project = load_project()
                threading.Thread(target=run_render, args=(project,), daemon=True).start()
                return self.send(200, {'ok': True})
        except Exception as e:
            traceback.print_exc()
            return self.send(500, {'error': str(e)})
        self.send(404, {'error': 'not found'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8765))
    srv = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    url = f'http://127.0.0.1:{port}'
    print(f'影片工作室已启动：{url}  （Ctrl+C 退出）')
    if '--no-browser' not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    srv.serve_forever()
