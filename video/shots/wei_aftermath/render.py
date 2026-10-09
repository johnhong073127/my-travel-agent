"""三国 · 战后（魏）5 秒 9:16 镜头：单张原画 → 2.5D 推镜 + 环境动态 + 音效。

python3 render.py <原画.png> <魏旗字体.ttf> <out.mp4> [--vo 旁白.wav] [--preview]
"""
import argparse
import math
import subprocess
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OW, OH, FPS, DUR = 1080, 1920, 30, 5.0
NF = int(FPS * DUR)
rng = np.random.default_rng(7)

ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('font'); ap.add_argument('out')
ap.add_argument('--vo', help='旁白 wav（单声道或立体声），从 VO_START 秒开始')
ap.add_argument('--vo-start', type=float, default=1.0)
ap.add_argument('--preview', action='store_true', help='只渲染 4 张关键帧')
args = ap.parse_args()

src = cv2.imread(args.src, cv2.IMREAD_COLOR).astype(np.float32) / 255  # BGR
SH, SW = src.shape[:2]
YY, XX = np.mgrid[0:SH, 0:SW].astype(np.float32)


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def fbm(h, w, scale, octaves=4, seed=0):
    r = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32); amp = 1.0; tot = 0
    for o in range(octaves):
        s = max(2, int(scale / 2 ** o))
        g = r.random((h // s + 3, w // s + 3)).astype(np.float32)
        out += amp * cv2.resize(g, (w + 3 * s, h + 3 * s), interpolation=cv2.INTER_CUBIC)[:h, :w]
        tot += amp; amp *= 0.5
    out /= tot
    return (out - out.min()) / (out.max() - out.min())


def poly_mask(polys, blur):
    m = np.zeros((SH, SW), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.array(p, np.int32)], 255)
    return cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), blur)


# ---------- 镜头：低机位前景 → 推进并升高 → 推向城门，最后约 0.8 秒近乎静止 ----------
ts = np.linspace(0, DUR, 2001)
vel = (0.45 + 0.55 * smooth(1.3, 2.0, ts)) * (1 - 0.93 * smooth(3.4, 4.25, ts))
prog = np.concatenate([[0], np.cumsum((vel[1:] + vel[:-1]) / 2 * np.diff(ts))]); prog /= prog[-1]


def camera(t):
    p = float(np.interp(t, ts, prog))
    h = lerp(900, 600, p)
    cx = lerp(420, 815, smooth(0, 1, p))
    cy = lerp(574, 482, smooth(0.15, 0.9, p))
    cy = min(cy, SH - h / 2)
    return p, cx, cy, h


FOCUS = np.array([822, 478], np.float32)          # 城门
PARALLAX, RISE = 0.28, 55

# 分层：前景大戟与残旗抠成独立图层整体平移缩放；背景层修补被遮挡处，按地面景深做视差
FG_DEPTH = 0.9
FG_POLYS = [
    [(203, 40), (230, 40), (372, 552), (346, 556)],                    # 戟杆
    [(300, 552), (482, 552), (505, 962), (392, 966)],                  # 戟刃
    [(0, 120), (245, 115), (318, 300), (255, 520), (120, 730), (0, 770)],  # 残旗
]
fg_alpha = poly_mask(FG_POLYS, 1.5)
hole = cv2.dilate((fg_alpha > 0.02).astype(np.uint8), np.ones((9, 9), np.uint8))
bg_src = cv2.inpaint((src * 255).astype(np.uint8), hole, 9, cv2.INPAINT_TELEA).astype(np.float32) / 255
for poly, shift in zip(FG_POLYS[:2], (42, 215)):        # 戟杆、戟刃：从右侧平移拷贝真实纹理，避免修补处发糊
    m = cv2.GaussianBlur(cv2.dilate(poly_mask([poly], 0.1).__gt__(0.5).astype(np.uint8), np.ones((13, 13), np.uint8)).astype(np.float32), (0, 0), 3)
    shifted = np.roll(src, -shift, axis=1)
    bg_src = lerp(bg_src, shifted, m[..., None])
fg_rgba = np.dstack([src, fg_alpha]).astype(np.float32)
depth = (np.clip((YY - 490) / (SH - 490), 0, 1) ** 1.2).astype(np.float32)

# 前景残旗：布面随风摆动（离旗杆越远摆幅越大）
cloth = poly_mask([[(0, 135), (225, 120), (300, 200), (300, 330), (230, 520), (130, 720), (0, 760)]], 10)
pole_x = 215 + (YY - 45) * (225 / 905)
cloth_amp = cloth * np.clip((pole_x - XX) / 260, 0, 1) * 7

# 火光：橙色高亮像素
b, g, r = src[..., 0], src[..., 1], src[..., 2]
fire = np.clip((r - b - 0.22) * 4, 0, 1) * np.clip((r - 0.5) * 3, 0, 1) * (r > g + 0.06)
fire = cv2.GaussianBlur(fire.astype(np.float32), (0, 0), 2)
flick_fields = [cv2.GaussianBlur(rng.random((SH // 16, SW // 16)).astype(np.float32), (0, 0), 1) for _ in range(24)]
flick_fields = [cv2.resize(f, (SW, SH), interpolation=cv2.INTER_CUBIC) for f in flick_fields]

# 天光：云层裂缝处的金色光
sky_glow = np.exp(-(((XX - 720) / 330) ** 2 + ((YY - 130) / 210) ** 2)) * (1 - smooth(260, 420, YY))
lum = src.mean(2)
WARM = np.array([0.50, 0.80, 1.0], np.float32)

# 烟雾：城前雾带 + 火点上升烟柱
FOG = fbm(SH + 200, SW + 300, 180, seed=3)
PLUME = fbm(SH + 400, SW + 100, 70, seed=4)
fog_band = np.exp(-((YY - 475) / 95) ** 2) * 0.32 + np.exp(-((YY - 600) / 120) ** 2) * 0.12
plume_mask = np.zeros((SH, SW), np.float32)
for px, py, wdt in [(650, 395, 45), (960, 395, 55), (1060, 330, 70), (590, 360, 40)]:
    up = np.clip((py - YY) / 220, 0, 1)
    plume_mask += np.exp(-((XX - px - up * 60) / (wdt * (1 + 2 * up))) ** 2) * (YY < py + 10) * (1 - up) ** 0.8
plume_mask = np.clip(plume_mask, 0, 1)
FOG_COL = np.array([0.47, 0.46, 0.46], np.float32)
SMOKE_COL = np.array([0.66, 0.66, 0.67], np.float32)


# ---------- 魏旗 ----------
def flag_texture(font_path, w=240, h=300):
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    body, edge = (30, 58, 112, 255), (16, 30, 66, 255)
    d.rectangle([0, 0, w - 1, h - 1], fill=edge)
    d.rectangle([14, 14, w - 26, h - 30], fill=body)
    for i in range(0, h, 26):                            # 旗尾与下摆锯齿边
        d.polygon([(w - 26, i), (w - 1, i + 13), (w - 26, i + 26)], fill=edge)
    for i in range(0, w - 20, 26):
        d.polygon([(i, h - 30), (i + 13, h - 1), (i + 26, h - 30)], fill=edge)
    font = ImageFont.truetype(font_path, 190)
    d.text(((w - 12) / 2, (h - 16) / 2), '魏', font=font, fill=(222, 204, 150, 255), anchor='mm')
    a = np.asarray(im).astype(np.float32) / 255
    rgb = a[..., [2, 1, 0]]                              # → BGR
    grain = fbm(h, w, 30, seed=11)[..., None]
    rgb *= 0.72 + 0.34 * grain                           # 布纹与污渍
    rgb *= 1 - 0.35 * smooth(0.55, 0.9, fbm(h, w, 70, seed=13))[..., None]  # 烟熏
    wear = fbm(h, w, 40, seed=12)
    edge_dist = np.minimum.outer(np.full(h, 1e9), np.arange(w)[::-1]).astype(np.float32)
    edge_dist = np.minimum(edge_dist, (h - 1 - np.arange(h))[:, None].astype(np.float32))
    alpha = a[..., 3] * (wear > 0.8 * np.exp(-edge_dist / 24))     # 只在旗尾与下摆破损
    return np.dstack([rgb, alpha]).astype(np.float32)


TEX = flag_texture(args.font)
TH, TW = TEX.shape[:2]
# (旗杆底 x, y, 旗杆高, 旗宽, 旗高, 雾化, 模糊 σ, 相位, 杆粗)
FLAGS = [
    (692, 652, 145, 45, 56, 0.14, 0.35, 0.0, 2),   # 清晰 A
    (902, 662, 150, 44, 55, 0.14, 0.35, 1.7, 2),   # 清晰 B
    (838, 338, 40, 22, 27, 0.30, 0.5, 3.1, 1),     # 城楼屋脊，衬着天空
    (612, 402, 26, 13, 16, 0.50, 0.9, 0.6, 1),     # 远景虚化
    (676, 404, 26, 13, 16, 0.50, 0.9, 2.2, 1),
    (948, 404, 26, 13, 16, 0.50, 0.9, 4.0, 1),
    (1002, 406, 26, 13, 16, 0.50, 0.9, 5.1, 1),
]


def waved_flag(t, phase, flutter):
    amp = TH * 0.13
    pad = int(amp * 2 + TH * 0.08)
    u = np.arange(TW, dtype=np.float32)[None, :].repeat(TH + pad, 0)
    v = np.arange(TH + pad, dtype=np.float32)[:, None].repeat(TW, 1)
    xf = u / TW
    ph = 2 * np.pi * (xf / 0.85 - flutter * t) + phase
    dy = amp * xf ** 1.2 * np.sin(ph) + TH * 0.07 * xf ** 2
    mx, my = u * (1 - 0.03 * xf * (1 + np.cos(ph))), v - dy - pad * 0.15
    out = cv2.remap(TEX, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    shade = 1 + 0.45 * np.cos(ph) * xf ** 0.8
    out[..., :3] *= shade[..., None]
    return out, pad * 0.15


def draw_flags(img, t):
    for fx, fy, ph_, fw, fh, haze, sig, phase, th in FLAGS:
        top = (fx + 3, fy - ph_)
        col = lerp(np.array([0.09, 0.09, 0.10]), FOG_COL, haze)
        cv2.line(img, (fx * 4, fy * 4), (top[0] * 4, top[1] * 4), col.tolist(), th, cv2.LINE_AA, shift=2)
        cv2.line(img, (top[0] * 4, top[1] * 4), (top[0] * 4, (top[1] - max(3, ph_ // 18)) * 4),
                 lerp(np.array([0.55, 0.55, 0.56]), FOG_COL, haze).tolist(), 1, cv2.LINE_AA, shift=2)
        fl, off = waved_flag(t, phase, 0.55 + 0.1 * math.sin(phase))
        sc = fw / TW
        fl = cv2.resize(fl, (fw, max(2, int(fl.shape[0] * sc))), interpolation=cv2.INTER_AREA)
        if sig > 0.4:
            fl = cv2.GaussianBlur(fl, (0, 0), sig)
        x0, y0 = top[0] + 1, int(top[1] + 2 - off * sc)
        hh, ww = fl.shape[:2]
        a = fl[..., 3:4]
        rgb = fl[..., :3] * 0.52                          # 阴天压暗，贴合原画曝光
        rgb = lerp(rgb.mean(2, keepdims=True), rgb, 0.8)  # 降饱和
        rgb = lerp(rgb, FOG_COL, haze)
        roi = img[y0:y0 + hh, x0:x0 + ww]
        roi[:] = roi * (1 - a) + rgb * a


# ---------- 输出空间：浮尘、余烬 ----------
N_DUST = 170
dust = np.column_stack([rng.random(N_DUST) * OW, rng.random(N_DUST) * OH, rng.random(N_DUST) ** 1.6])
dust_v = np.column_stack([10 + 35 * dust[:, 2] + rng.normal(0, 4, N_DUST), rng.normal(-2, 4, N_DUST)])
N_EMB = 36
emb_src = np.array([(650, 395), (960, 395), (1060, 335), (745, 525), (540, 535)], np.float32)
emb = np.column_stack([emb_src[rng.integers(0, len(emb_src), N_EMB)] + rng.normal(0, 12, (N_EMB, 2)),
                       rng.random(N_EMB) * 4, 18 + rng.random(N_EMB) * 25])
GROUND_DUST = fbm(OH + 200, OW + 400, 260, seed=9)
vy, vx = np.mgrid[0:OH, 0:OW].astype(np.float32)
vign = 1 - 0.38 * np.clip(np.hypot((vx - OW / 2) / (OW * 0.62), (vy - OH * 0.47) / (OH * 0.6)), 0, 1.4) ** 2.4
gu = ((np.arange(OW) + 0.5) / OW - 0.5).astype(np.float32)[None, :].repeat(OH, 0)
gv = ((np.arange(OH) + 0.5) / OH - 0.5).astype(np.float32)[:, None].repeat(OW, 1)


def to_out(x, y, cam):
    p, cx, cy, h = cam
    w = h * OW / OH
    return (x - cx) / w * OW + OW / 2, (y - cy) / h * OH + OH / 2


def render(i):
    t = i / FPS
    cam = camera(t)
    p, cx, cy, h = cam
    w = h * OW / OH
    img = bg_src.copy()

    # 火光闪烁 + 光晕
    k = t * 5; i0 = int(k) % len(flick_fields); fr = k - int(k)
    fl = lerp(flick_fields[i0], flick_fields[(i0 + 1) % len(flick_fields)], fr * fr * (3 - 2 * fr))
    flick = 0.7 + 0.6 * fl
    img *= (1 + fire * (flick - 1))[..., None]
    img += cv2.GaussianBlur(img * fire[..., None], (0, 0), 9) * (0.55 * flick)[..., None]

    # 金色天光逐渐透出
    gl = (0.05 + 0.16 * smooth(0, DUR, t)) * sky_glow * np.clip(lum - 0.25, 0, 1) * 2.2
    img += gl[..., None] * WARM

    # 烟雾流动
    fx_, fy_ = int(12 * t * 4) // 4, int(4 * t)
    fog = FOG[100 - fy_:100 - fy_ + SH, 150 - int(14 * t):150 - int(14 * t) + SW]
    img = lerp(img, FOG_COL, (fog_band * smooth(0.35, 0.85, fog))[..., None])
    py = int(26 * t)
    pl = PLUME[200 + py:200 + py + SH, 50 - int(8 * t):50 - int(8 * t) + SW]
    img = lerp(img, SMOKE_COL, (0.5 * plume_mask * smooth(0.4, 0.8, pl))[..., None])

    draw_flags(img, t)

    # 2.5D 推镜采样
    sx0, sy0 = (cx + gu * w).astype(np.float32), (cy + gv * h).astype(np.float32)
    d = cv2.remap(depth, sx0, sy0, cv2.INTER_LINEAR)
    s = 1 + PARALLAX * p * d
    sx, sy = FOCUS[0] + (sx0 - FOCUS[0]) / s, FOCUS[1] + (sy0 - FOCUS[1]) / s - RISE * p * d
    out = cv2.remap(img.astype(np.float32), sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    sf = 1 + PARALLAX * p * FG_DEPTH
    fx, fy = FOCUS[0] + (sx0 - FOCUS[0]) / sf, FOCUS[1] + (sy0 - FOCUS[1]) / sf - RISE * p * FG_DEPTH
    ca = cv2.remap(cloth_amp.astype(np.float32), fx, fy, cv2.INTER_LINEAR)
    fx = fx + ca * np.sin(0.024 * fy - 2 * np.pi * 0.7 * t + 0.012 * fx)
    fy = fy + 0.5 * ca * np.cos(0.02 * fx - 2 * np.pi * 0.45 * t)
    fgo = cv2.remap(fg_rgba, fx.astype(np.float32), fy.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a = np.clip(fgo[..., 3:4], 0, 1)
    out = out * (1 - a) + fgo[..., :3] * a

    # 地面浮尘雾
    gd = GROUND_DUST[100:100 + OH, 200 - int(30 * t):200 - int(30 * t) + OW]
    out = lerp(out, np.array([0.42, 0.44, 0.47], np.float32), (0.10 * smooth(0.45, 0.9, gd) * smooth(0.35, 0.8, vy / OH))[..., None])

    # 余烬（远景，随镜头变换）
    lay = np.zeros((OH, OW, 3), np.float32)
    for ex, ey, eph, evy in emb:
        yy = ey - evy * t; xx = ex + 6 * math.sin(eph + t * 2)
        ox, oy = to_out(xx, yy, cam)
        if 0 <= ox < OW and 0 <= oy < OH:
            a = 0.5 + 0.5 * math.sin(eph * 3 + t * 9)
            cv2.circle(lay, (int(ox * 4), int(oy * 4)), 6, (0.25 * a, 0.55 * a, 1.0 * a), -1, cv2.LINE_AA, shift=2)
    out += cv2.GaussianBlur(lay, (0, 0), 1.2) * 0.9 + cv2.GaussianBlur(lay, (0, 0), 6) * 0.6

    # 浮尘（随镜头前推向外扩散）
    camv = float(np.interp(t, ts, vel))
    lay = np.zeros((OH, OW), np.float32); lay_n = np.zeros((OH, OW), np.float32)
    fo = to_out(*FOCUS, cam)
    for j, (x0, y0, z) in enumerate(dust):
        dx, dy = x0 - fo[0], y0 - fo[1]
        x = (x0 + dust_v[j, 0] * t + dx * 0.12 * z * camv * t) % OW
        y = (y0 + dust_v[j, 1] * t + 4 * math.sin(t + j) + dy * 0.12 * z * camv * t) % OH
        tgt = lay_n if z > 0.6 else lay
        cv2.circle(tgt, (int(x * 4), int(y * 4)), int((1 + 5 * z) * 4), 0.10 + 0.30 * (1 - z) + 0.2 * z, -1, cv2.LINE_AA, shift=2)
    dl = cv2.GaussianBlur(lay, (0, 0), 0.8) + cv2.GaussianBlur(lay_n, (0, 0), 4) * 0.6
    out += dl[..., None] * np.array([0.62, 0.70, 0.78], np.float32) * 0.55

    # 调色：暗部偏青、亮部偏暖，暗角，锐化，颗粒
    l = out.mean(2, keepdims=True)
    out += (1 - l) ** 3 * np.array([0.018, 0.008, -0.004]) + l ** 3 * np.array([-0.02, 0.0, 0.025])
    out *= vign[..., None]
    out = out + 0.55 * (out - cv2.GaussianBlur(out, (0, 0), 1.6))
    out += np.random.default_rng(i).normal(0, 0.016, (OH, OW, 1)).astype(np.float32)
    return (np.clip(out, 0, 1) ** 1.03 * 255 + 0.5).astype(np.uint8)


# ---------- 音效：风 + 旗帜拍打 + 远处火声 + 低频铺底，最后一秒收成寂静 ----------
def sound(path, vo=None, vo_start=1.0):
    sr = 48000; n = int(sr * DUR); t = np.arange(n) / sr
    r = np.random.default_rng(5)

    def band(x, lo, hi):
        f = np.fft.rfftfreq(len(x), 1 / sr); X = np.fft.rfft(x)
        X *= (f > lo) * (f < hi) / np.sqrt(np.maximum(f, 20) / 100)
        return np.fft.irfft(X, len(x))

    def norm(x): return x / (np.abs(x).max() + 1e-9)
    hush = 1 - 0.75 * smooth(4.1, 4.8, t)
    gust = 0.6 + 0.4 * np.interp(t, np.linspace(0, DUR, 9), r.random(9))
    wind = np.stack([norm(band(r.normal(size=n), 80, 1400)) for _ in range(2)]) * gust * 0.22 * (0.6 + 0.4 * hush)
    flap_env = np.clip(np.sin(2 * np.pi * 2.6 * t + 0.6 * np.sin(2 * np.pi * 0.7 * t)), 0, 1) ** 3
    flap = norm(band(r.normal(size=n), 300, 2500)) * flap_env * 0.11 * (1 - smooth(1.8, 3.2, t)) * hush
    crack = np.zeros(n)
    for k in r.integers(0, n - 2000, 90):
        crack[k:k + 400] += r.normal(size=400) * np.exp(-np.arange(400) / 60) * r.random()
    crack = norm(band(crack, 400, 3500)) * 0.07 * (0.5 + 0.5 * smooth(1, 4, t)) * hush
    drone = sum(a * np.sin(2 * np.pi * f * t + 0.3 * np.sin(2 * np.pi * 0.11 * t))
                for f, a in [(55, 1), (82.4, 0.5), (110.3, 0.35), (164.8, 0.12)])
    drone = np.tanh(drone * 0.9) * 0.2 * smooth(0, 1.6, t) * (1 + 0.35 * smooth(2.5, 3.8, t)) * (1 - 0.85 * smooth(3.9, 4.6, t))
    L = wind[0] + flap * 1.0 + crack * 0.8 + drone
    R = wind[1] + flap * 0.5 + crack * 1.0 + drone
    mix = np.stack([L, R])
    if vo is not None:
        st = int(vo_start * sr); seg = vo[:, :n - st]
        mix[:, st:st + seg.shape[1]] += seg * 0.9
        mix[:, st:st + seg.shape[1]] *= 1                  # 旁白不压低环境声，靠电平取舍
    mix *= np.minimum(1, (n - np.arange(n)) / (0.05 * sr))
    mix = mix / max(1, np.abs(mix).max() / 0.89)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((mix.T * 32767).astype(np.int16).tobytes())


def read_wav(path):
    with wave.open(path) as wf:
        sr, ch = wf.getframerate(), wf.getnchannels()
        x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768
    x = x.reshape(-1, ch).T
    if ch == 1: x = np.repeat(x, 2, 0)
    assert sr == 48000, '旁白请先转成 48kHz'
    return x


if args.preview:
    for i in [0, 45, 105, 149]:
        cv2.imwrite(f'{args.out}_{i:03d}.jpg', render(i), [cv2.IMWRITE_JPEG_QUALITY, 90])
    raise SystemExit

wav = args.out + '.wav'
sound(wav, read_wav(args.vo) if args.vo else None, args.vo_start)
ff = subprocess.Popen([
    'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-',
    '-i', wav, '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-profile:v', 'high',
    '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', args.out,
], stdin=subprocess.PIPE)
for i in range(NF):
    ff.stdin.write(render(i).tobytes())
    if i % 15 == 0:
        print(f'\r{i}/{NF}', end='', flush=True)
ff.stdin.close(); ff.wait()
print(f'\r完成 {args.out}')
