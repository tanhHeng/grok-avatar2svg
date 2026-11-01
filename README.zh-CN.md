# grok-avatar2svg

[English](README.md) | **简体中文**

将 Grok 机器人风格的栅格头像转换为透明背景 SVG，保留平滑的色块和可变宽度的轮廓线。

输入是由色块和细线组成的栅格图像，例如 `example.png`。输出的 SVG 具有以下特点：

- 色块边界平滑且没有缝隙：两个区域之间的每条边只拟合一次，由两侧共用，因此相邻区域不会出现空隙或重叠。
- 将细线绘制为可变宽度的轮廓形状，末端可以渐细或平切，而不是使用固定宽度的描边。
- 背景保持透明。
- 可选择为颜色渐变的区域填充线性渐变。

照片、纹理和绘画式明暗效果不在处理范围内。

## 环境要求

- Python 3.11 或更新版本
- numpy、opencv-python、pillow、scikit-image、scipy
- resvg-py（可选），仅在使用 `--validate` 时需要

## 安装

```bash
pip install -e .              # 核心依赖
pip install -e ".[validate]"  # 同时安装用于验证的 resvg-py
```

如果 pip 安装到了用户目录，请确保对应的 Scripts 目录已加入 `PATH`，以便找到 `grok-avatar2svg` 命令。

## 使用方法

```bash
grok-avatar2svg example.png -o out/example.svg
```

安装后的命令行入口为 `grok-avatar2svg`。也可以运行 `python -m grok_avatar2svg`，参数相同。

| 选项 | 说明 |
| --- | --- |
| `-o`、`--output` | 输出路径。默认使用输入路径，并将后缀替换为 `.svg`。 |
| `--gradients` | 对区域内颜色变化明显的色块输出线性渐变。默认关闭。 |
| `--validate` | 用 resvg 将 SVG 渲染回图像，叠加到检测出的背景色上，并输出相对于输入图像的误差指标。 |

命令会输出 JSON 报告，其中包含数量统计（区域、笔画、边界线、曲线段、交汇点类型）、各阶段耗时，以及启用 `--validate` 时的误差指标。

在 Python 中调用：

```python
from pathlib import Path

from grok_avatar2svg.params import Params
from grok_avatar2svg.pipeline import load_image, vectorize

rgb, opaque = load_image("example.png")
result = vectorize(rgb, opaque, Params(gradients=True))
Path("example.svg").write_text(result.svg, encoding="utf-8")
print(result.stats)
```

所有阈值都在 `grok_avatar2svg/params.py` 中。像素值以 1254 px 图像为基准，并随输入尺寸缩放。

## 工作原理

1. **分割。** 以纯色色块和细线的种子为标记，通过分水岭算法扩展区域，再按颜色合并。若边界像素的颜色明显更接近相邻区域，就将其划入该区域，从而消除一个像素宽的碎条。
2. **背景。** 与图像边缘连通且颜色与边缘一致的区域变为透明。相同颜色的封闭区域（例如眼睛）则予以保留。
3. **线条。** 没有平坦核心的细长区域会转为笔画；较宽区域中的细长部分也一样，例如与同色色块合并的轮廓线。处理过程包括构建骨架图、提取在交叉处直行的链，以及平滑中心线。宽度和位置根据原图中线条的墨迹测量，因此较淡的部分会更细。自由端沿原图中的线条延伸，直到线条淡出为渐细尖端，或与另一笔画相遇；这可以重新连接被分割过程切断的线条。被遮挡物截断的末端则平切。
4. **边界图。** 区域边界构成由共享边组成的平面图。交汇点分为 T、Y 或 X 型，穿过同一交汇点的连续边共用一条切线。
5. **遮挡连续性。** 被前景物体（例如发夹）截断的轮廓会跨越遮挡物配对，并作为一条曲线拟合。
6. **曲线拟合。** 以亚像素精度定位边缘，在角点之间进行平滑，再用三次贝塞尔曲线拟合（Schneider 算法）。
7. **边界线。** 如果一条低对比度轮廓线在分割时被划入两侧区域，那么沿共享边会出现无法由两侧颜色通过抗锯齿混合得到的颜色。这些片段也会转为笔画。
8. **输出。** 区域填充下方铺设一层经过裁剪的底层，用于遮住抗锯齿缝隙，最后绘制笔画。

完整算法及各参数背后的测量依据见 [design.zh-CN.md](design.zh-CN.md)。

## `example.png` 的处理结果

| 指标 | 纯色 | `--gradients` |
| --- | --- | --- |
| 平均颜色误差（ΔE） | 1.08 | 0.99 |
| ΔE 的第 99 百分位数 | 7.43 | 5.44 |
| 平均边缘偏移 | 0.32 px | 0.30 px |
| 三次曲线段数 | 2787 | 2787 |
| 文件大小 | 82 KB | 84 KB |

处理一张 1254 × 1254 图像，完整运行约需 8 秒。

## 测试

```bash
python -m unittest discover -s tests
```

`tests/test_svgout.py` 中的接缝测试需要 resvg-py；未安装时会跳过这些测试。

## 项目结构

| 模块 | 职责 |
| --- | --- |
| `segment.py` | 种子、分水岭、区域合并、边界像素清理 |
| `strokes.py` | 线条检测、边界线和可变宽度笔画轮廓 |
| `background.py` | 背景检测 |
| `planar.py` | 共享边界边的平面图 |
| `subpixel.py` | 亚像素边缘定位 |
| `junctions.py` | 交汇点分类和端点切线 |
| `occlusion.py` | 跨遮挡物的轮廓配对 |
| `fitting.py` | 带平滑和强制断点的贝塞尔拟合 |
| `coloring.py` | 区域颜色和可选渐变 |
| `svgout.py` | SVG 组装 |
| `validate.py` | 回渲染误差指标 |
| `pipeline.py`、`__main__.py` | 流程编排和命令行接口 |

## 局限

- 参数仅针对一张示例图像调校。
- 线条检测依赖抗锯齿：没有抗锯齿的线条具有平坦核心，会被当作色块处理。
- 输出针对保真度和体积优化，而非编辑便利性。被遮挡形状的隐藏部分不会重建。
