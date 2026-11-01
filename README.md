# grok-avatar2svg

Convert Grok bot-style raster avatars into transparent SVGs with smooth fills and variable-width outlines.

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

If pip installs into the user site, ensure its Scripts directory is on `PATH` so the `grok-avatar2svg` command can be found.

## Usage

```bash
grok-avatar2svg example.png -o out/example.svg
```

The CLI is installed as `grok-avatar2svg`. Alternatively, run `python -m grok_avatar2svg` with the same arguments.

| Option | Description |
| --- | --- |
| `-o`, `--output` | Output path. Defaults to the input path with an `.svg` suffix. |
| `--gradients` | Emit linear gradients for regions whose color changes noticeably across them. Off by default. |
| `--validate` | Render the SVG back with resvg, composite it over the detected background color, and print error metrics against the input. |

The command prints a JSON report with counts (regions, strokes, boundary lines, curve segments, junction types), per-stage timings, and, with `--validate`, the error metrics.

From Python:

```python
from pathlib import Path

from grok_avatar2svg.params import Params
from grok_avatar2svg.pipeline import load_image, vectorize

rgb, opaque = load_image("example.png")
result = vectorize(rgb, opaque, Params(gradients=True))
Path("example.svg").write_text(result.svg, encoding="utf-8")
print(result.stats)
```

Every threshold lives in `grok_avatar2svg/params.py`. Pixel values are given for a 1254 px image and scale with the input size.

## How it works

1. **Segmentation.** Flat-color seeds plus thin-line seeds are grown with a marker-based watershed, then merged by color. Boundary pixels that clearly match a neighboring region's color are moved to it, which removes one-pixel slivers.
2. **Background.** Regions connected to the image border with the border color become transparent. Enclosed regions of the same color, such as eyes, are kept.
3. **Lines.** Thin, elongated regions without a flat core become strokes, and so do thin parts of wider regions, such as an outline that merged with a block of its own color: skeleton graph, chains that pass straight through crossings, and a smoothed centerline. Width and position are measured from the line's ink in the image, so faint stretches come out thin. Free ends follow the line in the image until it fades into a tapered tip or meets another stroke, which rejoins lines that segmentation broke apart; ends at an occluder are cut flat.
4. **Boundary graph.** Region boundaries form a planar map of shared edges. Junctions are classified as T, Y or X, and edges that continue through a junction share one tangent.
5. **Occlusion continuity.** Contours cut by a front object, such as a hair clip, are paired across it and fitted as one curve.
6. **Curve fitting.** Edges are placed at subpixel accuracy, smoothed between corners, and fitted with cubic Bezier curves (Schneider's algorithm).
7. **Boundary lines.** A low-contrast outline that segmentation split between the two regions beside it shows up along their shared edge as colors that no anti-aliasing of the two sides can produce. Such stretches become strokes too.
8. **Output.** Region fills sit on top of a clipped underlay that hides anti-aliasing seams, and strokes are drawn last.

The full algorithm, with the measurements behind each parameter, is in [design.md](design.md) (written in Chinese).

## Results on `example.png`

| Metric | Flat colors | `--gradients` |
| --- | --- | --- |
| Mean color error (ΔE) | 1.08 | 0.99 |
| 99th percentile ΔE | 7.43 | 5.44 |
| Mean edge offset | 0.32 px | 0.30 px |
| Cubic segments | 2787 | 2787 |
| File size | 82 KB | 84 KB |

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
| `strokes.py` | Line detection, boundary lines and variable-width stroke outlines |
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
