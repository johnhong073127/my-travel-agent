// 逐帧离线渲染：node render.mjs <spec.json> [out.mp4]
// 输出 1080x1920 / 30fps / H.264 High + AAC / faststart，可直接上传 YouTube Shorts 与小红书
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const [specPath, outArg] = process.argv.slice(2);
if (!specPath) {
  console.error('用法: node render.mjs <spec.json> [out.mp4]');
  process.exit(1);
}
const here = path.dirname(fileURLToPath(import.meta.url));
const spec = JSON.parse(fs.readFileSync(specPath, 'utf8'));
const out = path.resolve(outArg || path.join(here, 'out', path.basename(specPath, '.json') + '.mp4'));
fs.mkdirSync(path.dirname(out), { recursive: true });

const MIME = { '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp' };
spec.images = (spec.images || []).map(p => {
  const abs = path.resolve(path.dirname(specPath), p);
  return `data:${MIME[path.extname(abs).toLowerCase()] || 'image/jpeg'};base64,${fs.readFileSync(abs).toString('base64')}`;
});

const browser = await chromium.launch(
  fs.existsSync('/opt/pw-browsers/chromium') ? { executablePath: '/opt/pw-browsers/chromium' } : {});
const page = await browser.newPage();
page.on('pageerror', e => console.error('页面错误:', e.message));
await page.goto(pathToFileURL(path.join(here, 'index.html')).href);
await page.evaluate(s => window.VideoKit.load(s), spec);
const { DURATION, FPS } = await page.evaluate(() => ({ DURATION: VideoKit.DURATION, FPS: VideoKit.FPS }));

const wav = path.join(os.tmpdir(), `music-${process.pid}.wav`);
const audioIn = spec.music === false
  ? ['-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo']
  : (fs.writeFileSync(wav, Buffer.from(await page.evaluate(() => VideoKit.music()), 'base64')), ['-i', wav]);

const ff = spawn('ffmpeg', [
  '-y', '-v', 'error',
  '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-',
  ...audioIn,
  '-t', String(DURATION),
  '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-r', String(FPS),
  '-c:a', 'aac', '-b:a', '192k', '-ar', '44100',
  '-movflags', '+faststart', out,
], { stdio: ['pipe', 'inherit', 'inherit'] });
const finished = new Promise((res, rej) => ff.on('close', code => code ? rej(new Error(`ffmpeg 退出码 ${code}`)) : res()));

const total = DURATION * FPS;
for (let i = 0; i < total; i++) {
  const url = await page.evaluate(t => VideoKit.frame(t), i / FPS);
  if (!ff.stdin.write(Buffer.from(url.slice(url.indexOf(',') + 1), 'base64'))) {
    await new Promise(r => ff.stdin.once('drain', r));
  }
  if (i % FPS === 0) process.stdout.write(`\r渲染 ${i}/${total}`);
}
ff.stdin.end();
await finished;
await browser.close();
fs.rmSync(wav, { force: true });
console.log(`\r完成: ${out}`);
