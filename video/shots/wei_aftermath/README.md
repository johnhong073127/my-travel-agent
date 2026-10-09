# 三国 · 战后（魏）— 5 秒 9:16 镜头

把一张横版原画做成 2.5D 电影镜头：低机位前景 → 推进并升高 → 推向城门，最后约 0.8 秒近乎静止。

- 前景大戟与残旗抠成独立图层，背景按地面景深做视差，被遮挡的部分从旁边拷贝纹理补上
- 程序化加入：残旗飘动、3 面清晰魏旗（中景 2 面 + 城楼 1 面）和 4 面虚化远旗、烟雾流动、火光闪烁、余烬、浮尘、金色天光渐亮
- 音效：风声、旗帜拍打、远处火声、低频铺底，最后一秒收成寂静

```
pip install opencv-python-headless numpy pillow
python3 render.py 原画.png MaShanZheng-Regular.ttf out.mp4 [--vo 旁白.wav --vo-start 1.0]
python3 render.py 原画.png MaShanZheng-Regular.ttf pv --preview   # 只出 4 张关键帧
```

字体：Google Fonts「Ma Shan Zheng」（OFL）。原画坐标按 1536×1024 标定，换图需要重新标定 `FG_POLYS`、`FLAGS`、`FOCUS`。
