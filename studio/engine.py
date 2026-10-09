"""渲染引擎：单张原画 → 2.5D 电影镜头；多镜头 → 转场拼接 + 旁白/环境声/配乐混音。"""
import math
import subprocess

import cv2
import numpy as np

OW, OH, FPS, SR = 1080, 1920, 30, 48000

DEFAULT_SHOT = {
    'image': None, 'duration': 5.0, 'transition': 0.5, 'hold': 0.8,
    'cam_start': {'x': 0.5, 'y': 0.6, 'zoom': 1.1},
    'cam_end': {'x': 0.5, 'y': 0.45, 'zoom': 1.6},
    'horizon': 0.5, 'parallax': 0.15, 'rise': 0.04,
    'fog': 0.5, 'dust': 0.5, 'embers': 0.5, 'fire': 0.6,
    'light': {'x': 0.5, 'y': 0.15, 'strength': 0.5},
    'grain': 0.5, 'ambience': 'battlefield', 'hush_end': True,
}


def shot_params(shot):
    p = {**DEFAULT_SHOT, **{k: v for k, v in shot.items() if v is not None}}
    for k in ('cam_start', 'cam_end', 'light'):
        p[k] = {**DEFAULT_SHOT[k], **(shot.get(k) or {})}
    return p


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def fbm(h, w, scale, octaves=4, seed=0):
    r = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32); amp = 1.0; tot = 0
    for _ in range(octaves):
        s = max(2, int(scale))
        g = r.random((h // s + 3, w // s + 3)).astype(np.float32)
        out += amp * cv2.resize(g, (w + 3 * s, h + 3 * s), interpolation=cv2.INTER_CUBIC)[:h, :w]
        tot += amp; amp *= 0.5; scale /= 2
    out /= tot
    return (out - out.min()) / (out.max() - out.min() + 1e-9)


class Shot:
    """一个镜头。scale < 1 用于快速预览。"""

    def __init__(self, image_path, shot, scale=1.0):
        self.p = p = shot_params(shot)
        self.ow, self.oh = int(OW * scale) // 2 * 2, int(OH * scale) // 2 * 2
        src = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if src is None:
            raise ValueError(f'无法读取图片 {image_path}')
        self.src = src.astype(np.float32) / 255
        SH, SW = self.SH, self.SW = self.src.shape[:2]
        self.D = float(p['duration'])
        self.rng = np.random.default_rng(7)
        YY, XX = np.mgrid[0:SH, 0:SW].astype(np.float32)

        # 运镜：缓入 → 推进 → 结尾 hold 秒近乎静止
        D, hold = self.D, min(float(p['hold']), self.D * 0.5)
        self.ts = ts = np.linspace(0, D, 1001)
        dm = D - hold
        self.vel = vel = (0.35 + 0.65 * smooth(0, 0.4 * dm, ts)) * (1 - 0.93 * smooth(dm - min(0.8, dm * 0.3), dm, ts))
        prog = np.concatenate([[0], np.cumsum((vel[1:] + vel[:-1]) / 2 * np.diff(ts))])
        self.prog = prog / prog[-1]
        self.full_h = min(SH, SW * 16 / 9)

        hz = p['horizon'] * SH
        self.depth = (np.clip((YY - hz) / (SH - hz + 1), 0, 1) ** 1.2).astype(np.float32)

        b, g, r = self.src[..., 0], self.src[..., 1], self.src[..., 2]
        fire = np.clip((r - b - 0.22) * 4, 0, 1) * np.clip((r - 0.5) * 3, 0, 1) * (r > g + 0.06)
        self.fire = cv2.GaussianBlur(fire.astype(np.float32), (0, 0), 2)
        fs = max(8, SW // 96)
        self.flick = [cv2.resize(cv2.GaussianBlur(self.rng.random((SH // fs + 1, SW // fs + 1)).astype(np.float32), (0, 0), 1),
                                 (SW, SH), interpolation=cv2.INTER_CUBIC) for _ in range(20)]

        L = p['light']
        self.glow = (np.exp(-(((XX - L['x'] * SW) / (0.22 * SW)) ** 2 + ((YY - L['y'] * SH) / (0.2 * SH)) ** 2))
                     * np.clip(self.src.mean(2) - 0.25, 0, 1) * 2.2).astype(np.float32)

        self.fog_tex = fbm(SH + 120, SW + 400, SW / 8, seed=3)
        self.fog_band = (np.exp(-((YY - hz) / (0.1 * SH)) ** 2) * 0.32 + np.exp(-((YY - hz - 0.12 * SH) / (0.12 * SH)) ** 2) * 0.12).astype(np.float32)

        ys, xs = np.nonzero(self.fire > 0.4)
        n_emb = int(50 * p['embers']) if len(xs) else 0
        pick = self.rng.integers(0, len(xs), n_emb) if n_emb else []
        self.emb = [(xs[k] + self.rng.normal(0, 6), ys[k], self.rng.random() * 6, (0.015 + 0.025 * self.rng.random()) * SH)
                    for k in pick]

        ow, oh = self.ow, self.oh
        n = int(170 * p['dust'])
        self.dust = np.column_stack([self.rng.random(n) * ow, self.rng.random(n) * oh, self.rng.random(n) ** 1.6])
        self.dust_v = np.column_stack([(10 + 35 * self.dust[:, 2] + self.rng.normal(0, 4, n)) * scale, self.rng.normal(-2, 4, n) * scale])
        self.gdust = fbm(oh + 100, ow + 300, ow / 4, seed=9)
        vy, vx = np.mgrid[0:oh, 0:ow].astype(np.float32)
        self.vy = vy
        self.vign = (1 - 0.38 * np.clip(np.hypot((vx - ow / 2) / (ow * 0.62), (vy - oh * 0.47) / (oh * 0.6)), 0, 1.4) ** 2.4)[..., None]
        self.gu = ((np.arange(ow) + 0.5) / ow - 0.5).astype(np.float32)[None, :].repeat(oh, 0)
        self.gv = ((np.arange(oh) + 0.5) / oh - 0.5).astype(np.float32)[:, None].repeat(ow, 1)
        self.scale = scale

    def camera(self, t):
        p = float(np.interp(t, self.ts, self.prog))
        a, b = self.p['cam_start'], self.p['cam_end']
        h = self.full_h / lerp(max(a['zoom'], 1), max(b['zoom'], 1), p)
        w = h * 9 / 16
        e = smooth(0, 1, p)
        cx = np.clip(lerp(a['x'], b['x'], e) * self.SW, w / 2, self.SW - w / 2)
        cy = np.clip(lerp(a['y'], b['y'], smooth(0.1, 0.9, p)) * self.SH, h / 2, self.SH - h / 2)
        return p, float(cx), float(cy), h, w

    def frame(self, i):
        P, t = self.p, i / FPS
        SH, SW, ow, oh = self.SH, self.SW, self.ow, self.oh
        p, cx, cy, h, w = self.camera(t)
        img = self.src.copy()

        if P['fire'] > 0:
            k = t * 5; i0 = int(k) % len(self.flick); fr = k - int(k)
            fl = lerp(self.flick[i0], self.flick[(i0 + 1) % len(self.flick)], fr * fr * (3 - 2 * fr))
            flick = 1 + P['fire'] * (fl - 0.5) * 1.2
            img *= (1 + self.fire * (flick - 1))[..., None]
            img += cv2.GaussianBlur(img * self.fire[..., None], (0, 0), max(2, SW / 170)) * (0.9 * P['fire'] * flick)[..., None]

        if P['light']['strength'] > 0:
            img += (P['light']['strength'] * (0.1 + 0.3 * smooth(0, self.D, t)) * self.glow)[..., None] * np.array([0.5, 0.8, 1.0], np.float32)

        if P['fog'] > 0:
            dx, dy = int(0.01 * SW * t), int(0.004 * SH * t)
            f = self.fog_tex[60 - min(dy, 59):60 - min(dy, 59) + SH, 200 - min(dx, 199):200 - min(dx, 199) + SW]
            img = lerp(img, np.array([0.47, 0.46, 0.46], np.float32), (P['fog'] * 1.6 * self.fog_band * smooth(0.35, 0.85, f))[..., None])

        sx0 = (cx + self.gu * w).astype(np.float32); sy0 = (cy + self.gv * h).astype(np.float32)
        d = cv2.remap(self.depth, sx0, sy0, cv2.INTER_LINEAR)
        s = 1 + P['parallax'] * p * d
        fx, fy = self.p['cam_end']['x'] * SW, self.p['cam_end']['y'] * SH
        sx = (fx + (sx0 - fx) / s).astype(np.float32)
        sy = (fy + (sy0 - fy) / s - P['rise'] * SH * p * d).astype(np.float32)
        out = cv2.remap(img.astype(np.float32), sx, sy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        if P['dust'] > 0:
            gd = self.gdust[50:50 + oh, 300 - min(int(30 * self.scale * t), 299):][:, :ow]
            out = lerp(out, np.array([0.42, 0.44, 0.47], np.float32), (0.10 * P['dust'] * 2 * smooth(0.45, 0.9, gd) * smooth(0.35, 0.8, self.vy / oh))[..., None])

        if self.emb:
            lay = np.zeros((oh, ow, 3), np.float32)
            for ex, ey, eph, evy in self.emb:
                xx, yy = ex + 0.004 * SW * math.sin(eph + 2 * t), ey - evy * t
                ox, oy = (xx - cx) / w * ow + ow / 2, (yy - cy) / h * oh + oh / 2
                if 0 <= ox < ow and 0 <= oy < oh:
                    a = 0.5 + 0.5 * math.sin(eph * 3 + t * 9)
                    cv2.circle(lay, (int(ox * 4), int(oy * 4)), max(4, int(6 * self.scale)), (0.25 * a, 0.55 * a, a), -1, cv2.LINE_AA, shift=2)
            out += cv2.GaussianBlur(lay, (0, 0), 1.2 * self.scale) * 0.9 + cv2.GaussianBlur(lay, (0, 0), 6 * self.scale) * 0.6

        if len(self.dust):
            camv = float(np.interp(t, self.ts, self.vel))
            lay = np.zeros((oh, ow), np.float32); lay_n = np.zeros((oh, ow), np.float32)
            fo = ((fx - cx) / w * ow + ow / 2, (fy - cy) / h * oh + oh / 2)
            for j, (x0, y0, z) in enumerate(self.dust):
                x = (x0 + self.dust_v[j, 0] * t + (x0 - fo[0]) * 0.12 * z * camv * t) % ow
                y = (y0 + self.dust_v[j, 1] * t + 4 * self.scale * math.sin(t + j) + (y0 - fo[1]) * 0.12 * z * camv * t) % oh
                cv2.circle(lay_n if z > 0.6 else lay, (int(x * 4), int(y * 4)), max(4, int((1 + 5 * z) * 4 * self.scale)),
                           0.10 + 0.30 * (1 - z) + 0.2 * z, -1, cv2.LINE_AA, shift=2)
            dl = cv2.GaussianBlur(lay, (0, 0), 0.8) + cv2.GaussianBlur(lay_n, (0, 0), 4 * self.scale) * 0.6
            out += dl[..., None] * np.array([0.62, 0.70, 0.78], np.float32) * 0.55

        l = out.mean(2, keepdims=True)
        out += (1 - l) ** 3 * np.array([0.018, 0.008, -0.004]) + l ** 3 * np.array([-0.02, 0.0, 0.025])
        out *= self.vign
        out = out + 0.55 * (out - cv2.GaussianBlur(out, (0, 0), 1.6 * self.scale))
        if P['grain'] > 0:
            out += np.random.default_rng(i).normal(0, 0.03 * P['grain'], (oh, ow, 1)).astype(np.float32)
        return (np.clip(out, 0, 1) ** 1.03 * 255 + 0.5).astype(np.uint8)

    def render(self, out_path, progress=None):
        n = int(round(self.D * FPS))
        ff = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{self.ow}x{self.oh}',
                               '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '14',
                               '-pix_fmt', 'yuv420p', out_path], stdin=subprocess.PIPE)
        for i in range(n):
            ff.stdin.write(self.frame(i).tobytes())
            if progress:
                progress(i + 1, n)
        ff.stdin.close()
        if ff.wait():
            raise RuntimeError('ffmpeg 编码失败')


# ---------------- 音频 ----------------
def load_audio(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-f', 'f32le', '-ac', '2', '-ar', str(SR), '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy()


def audio_duration(path):
    return load_audio(path).shape[1] / SR


def ambience(kind, D, hush_end, seed):
    n = int(SR * D); t = np.arange(n) / SR
    if kind == 'none' or n == 0:
        return np.zeros((2, n), np.float32)
    r = np.random.default_rng(seed)

    def band(x, lo, hi):
        f = np.fft.rfftfreq(len(x), 1 / SR); X = np.fft.rfft(x)
        X *= (f > lo) * (f < hi) / np.sqrt(np.maximum(f, 20) / 100)
        return np.fft.irfft(X, len(x))

    def norm(x): return x / (np.abs(x).max() + 1e-9)
    hush = 1 - 0.75 * smooth(D - 0.9, D - 0.2, t) if hush_end else np.ones(n)
    gust = 0.6 + 0.4 * np.interp(t, np.linspace(0, D, 9), r.random(9))
    mix = np.stack([norm(band(r.normal(size=n), 80, 1400)) for _ in range(2)]) * gust * 0.22 * (0.6 + 0.4 * hush)
    if kind == 'battlefield':
        crack = np.zeros(n)
        for k in r.integers(0, max(1, n - 2000), int(18 * D)):
            crack[k:k + 400] += r.normal(size=len(crack[k:k + 400])) * np.exp(-np.arange(len(crack[k:k + 400])) / 60) * r.random()
        crack = norm(band(crack, 400, 3500)) * 0.07 * hush
        drone = sum(a * np.sin(2 * np.pi * f * t + 0.3 * np.sin(2 * np.pi * 0.11 * t)) for f, a in [(55, 1), (82.4, 0.5), (110.3, 0.35)])
        drone = np.tanh(drone * 0.9) * 0.18 * smooth(0, min(1.6, D / 3), t) * (1 - 0.85 * (1 - hush) / 0.75)
        mix = mix + np.stack([crack * 0.8 + drone, crack + drone])
    return mix.astype(np.float32)


def mix_audio(shots, offsets, narration, music, project_dir, total):
    n = int(SR * total) + 1
    out = np.zeros((2, n), np.float32)
    for k, (sh, off) in enumerate(zip(shots, offsets)):
        p = shot_params(sh)
        a = ambience(p['ambience'], p['duration'], p['hush_end'], k)
        tr = int(SR * p['transition'])
        if tr:
            a[:, :tr] *= np.linspace(0, 1, tr) if k else 1
            a[:, -tr:] *= np.linspace(1, 0, tr) if k < len(shots) - 1 else 1
        st = int(off * SR); out[:, st:st + a.shape[1]] += a[:, :n - st]
    vo = np.zeros(n, np.float32)
    for line in narration:
        if not line.get('audio'):
            continue
        a = load_audio(f"{project_dir}/{line['audio']}") * float(line.get('gain', 1.0))
        st = int(float(line.get('start', 0)) * SR)
        if st >= n:
            continue
        seg = a[:, :n - st]
        out[:, st:st + seg.shape[1]] += seg
        vo[st:st + seg.shape[1]] = 1
    if music and music.get('audio'):
        m = load_audio(f"{project_dir}/{music['audio']}")
        m = np.tile(m, (1, n // m.shape[1] + 1))[:, :n] * float(music.get('volume', 0.5))
        duck = 1 - float(music.get('duck', 0.6)) * np.clip(np.convolve(vo, np.ones(SR // 4) / (SR // 4), 'same') * 1.5, 0, 1)
        fade = np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / SR)
        out += m * duck * fade
    out *= np.minimum(1, (n - np.arange(n)) / (0.05 * SR))
    return out


def write_wav(path, x):
    import wave
    x = x / max(1, np.abs(x).max() / 0.89)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((x.T * 32767).astype(np.int16).tobytes())


def timeline(shots):
    """每个镜头在成片中的起点（转场与上一镜头重叠）及总时长。"""
    offsets, cur, prev = [], 0.0, None
    for sh in shots:
        p = shot_params(sh)
        if prev is not None:
            cur -= min(p['transition'], prev, p['duration'])
        offsets.append(cur)
        cur += p['duration']; prev = p['duration']
    return offsets, cur


def render_film(project, project_dir, out_path, progress=None):
    shots = [s for s in project['shots'] if s.get('image')]
    if not shots:
        raise ValueError('至少需要一个带图片的镜头')
    offsets, total = timeline(shots)
    clips = []
    for k, sh in enumerate(shots):
        clip = f'{project_dir}/cache/shot_{k}.mp4'
        Shot(f"{project_dir}/{sh['image']}", sh).render(
            clip, progress and (lambda i, n, k=k: progress(f'渲染镜头 {k + 1}/{len(shots)}', (k + i / n) / (len(shots) + 0.3))))
        clips.append(clip)
    if progress:
        progress('混音与合成', len(shots) / (len(shots) + 0.3))
    wav = f'{project_dir}/cache/mix.wav'
    write_wav(wav, mix_audio(shots, offsets, project.get('narration', []), project.get('music'), project_dir, total))

    args = ['ffmpeg', '-y', '-v', 'error']
    for c in clips:
        args += ['-i', c]
    args += ['-i', wav]
    graph, last = [], '[0:v]'
    for k in range(1, len(clips)):
        tr = round(offsets[k - 1] + shot_params(shots[k - 1])['duration'] - offsets[k], 3)
        lab = f'[v{k}]'
        if tr > 0:
            graph.append(f'{last}[{k}:v]xfade=transition=fade:duration={tr}:offset={offsets[k]:.3f}{lab}')
        else:
            graph.append(f'{last}[{k}:v]concat=n=2:v=1:a=0{lab}')
        last = lab
    vmap = last if graph else '0:v'
    if graph:
        args += ['-filter_complex', ';'.join(graph)]
    args += ['-map', vmap, '-map', f'{len(clips)}:a', '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11',
             '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-r', str(FPS),
             '-c:a', 'aac', '-b:a', '192k', '-ar', str(SR), '-t', f'{total:.3f}', '-movflags', '+faststart', out_path]
    subprocess.run(args, check=True)
    if progress:
        progress('完成', 1.0)
    return total
