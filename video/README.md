# 10 秒竖屏短视频生成器

面向 YouTube Shorts 与小红书：1080×1920、30fps、H.264 + AAC、10 秒。

- **出片（推荐）**：`node render.mjs specs/xxx.json` 逐帧离线渲染，画面稳定、编码兼容各平台。在 Claude Code 里用 `/make-video <主题>` 可自动完成写脚本 → 渲染 → 自检 → 交付。
- **手动预览**：浏览器打开 `index.html`，改文字和图片后可实时预览；页面里也能直接录制，但录制质量取决于浏览器，只适合预览。

依赖：Node 18+、ffmpeg（含 libx264）、`npm install`（Playwright Chromium）。
