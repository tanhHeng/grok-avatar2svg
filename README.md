# avatar2svg

Convert flat-color, Grok-bot-like AI avatars into compact SVG files built from filled paths.

The input is a raster image made of color blocks and thin lines, such as `example.png`. The output is an SVG that:

- has smooth region boundaries with no seams: every edge between two regions is fitted once and shared by both, so neighbors can never gap or overlap;
- draws thin lines as variable-width outlines with tapered or flat-cut ends, instead of fixed-width strokes;
- leaves the background transparent;
- optionally fills regions that contain a color ramp with a linear gradient.

Photos, textures and painterly shading are out of scope.

## Requirements

- Python 3.11 or newer
- numpy, opencv-python, pillow, scikit-image, scipy
- resvg-py (optional), needed only for `--validate`

## Installation

```bash
pip install -e .              # core dependencies
pip install -e ".[validate]"  # also install resvg-py for validation
```

## Usage

```bash
python -m avatar2svg example.png -o out/example.svg
```

| Option | Description |
| --- | --- |
| `-o`, `--output` | Output path. Defaults to the input path with an `.svg` suffix. |
| `--gradients` | Emit linear gradients for regions whose color changes noticeably across them. Off by default. |
| `--validate` | Render the SVG back with resvg, composite it over the detected background color, and print error metrics against the input. |

The command prints a JSON report with counts (regions, strokes, curve segments, junction types), per-stage timings, and, with `--validate`, the error metrics.

From Python:

```python
from pathlib import Path

from avatar2svg.params import Params
from avatar2svg.pipeline import load_image, vectorize

rgb, opaque = load_image("example.png")
result = vectorize(rgb, opaque, Params(gradients=True))
Path("example.svg").write_text(result.svg, encoding="utf-8")
print(result.stats)
```

Every threshold lives in `avatar2svg/params.py`. Pixel values are given for a 1254 px image and scale with the input size.

## How it works

1. **Segmentation.** Flat-color seeds plus thin-line seeds are grown with a marker-based watershed, then merged by color. Boundary pixels that clearly match a neighboring region's color are moved to it, which removes one-pixel slivers.
2. **Lines.** Thin, elongated regions without a flat core become strokes: skeleton graph, chains that pass straight through crossings, a smoothed centerline and width profile, and tapered, round or flat-cut ends.
3. **Background.** Regions connected to the image border with the border color become transparent. Enclosed regions of the same color, such as eyes, are kept.
4. **Boundary graph.** Region boundaries form a planar map of shared edges. Junctions are classified as T, Y or X, and edges that continue through a junction share one tangent.
5. **Occlusion continuity.** Contours cut by a front object, such as a hair clip, are paired across it and fitted as one curve.
6. **Curve fitting.** Edges are placed at subpixel accuracy, smoothed between corners, and fitted with cubic Bezier curves (Schneider's algorithm).
7. **Output.** Region fills sit on top of a clipped underlay that hides anti-aliasing seams, and strokes are drawn last.

The full algorithm, with the measurements behind each parameter, is in [design.md](design.md) (written in Chinese).

## Results on `example.png`

| Metric | Flat colors | `--gradients` |
| --- | --- | --- |
| Mean color error (ΔE) | 1.09 | 1.00 |
| 99th percentile ΔE | 7.55 | 5.84 |
| Mean edge offset | 0.32 px | 0.30 px |
| Cubic segments | 2575 | 2575 |
| File size | 75 KB | 78 KB |

A full run takes about 8 seconds on a 1254 × 1254 image.

## Tests

```bash
python -m unittest discover -s tests
```

The seam tests in `tests/test_svgout.py` need resvg-py and are skipped without it.

## Project layout

| Module | Role |
| --- | --- |
| `segment.py` | Seeds, watershed, region merging, boundary-pixel cleanup |
| `strokes.py` | Line detection and variable-width stroke outlines |
| `background.py` | Background detection |
| `planar.py` | Planar map of shared boundary edges |
| `subpixel.py` | Subpixel edge placement |
| `junctions.py` | Junction typing and end tangents |
| `occlusion.py` | Pairing contours across occluders |
| `fitting.py` | Bezier fitting with smoothing and forced breaks |
| `coloring.py` | Region colors and optional gradients |
| `svgout.py` | SVG assembly |
| `validate.py` | Render-back metrics |
| `pipeline.py`, `__main__.py` | Orchestration and command-line interface |

## Limitations

- Parameters were tuned on a single sample image.
- Line detection relies on anti-aliasing: a line drawn without it has a flat core and is treated as a color block.
- The output is optimized for fidelity and size, not for editing. Hidden parts of occluded shapes are not reconstructed.
