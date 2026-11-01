from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .params import Params
from .pipeline import load_image, vectorize


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="grok-avatar2svg", description="Convert a Grok bot-style raster avatar into transparent SVG.")
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="output SVG path (default: input with .svg suffix)")
    ap.add_argument("--gradients", action="store_true", help="emit linear gradients for regions that need them")
    ap.add_argument("--validate", action="store_true", help="rasterize the SVG back and print error metrics (needs resvg-py)")
    args = ap.parse_args(argv)

    rgb, opaque = load_image(args.input)
    result = vectorize(rgb, opaque, Params(gradients=args.gradients))
    out = args.output or args.input.with_suffix(".svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.svg, encoding="utf-8")

    report: dict[str, object] = {
        "output": str(out),
        "stats": result.stats,
        "timings_s": {k: round(v, 2) for k, v in result.timings.items()},
    }
    if args.validate:
        from .validate import metrics

        report["metrics"] = {
            k: round(v, 3) if isinstance(v, float) else v
            for k, v in metrics(result.svg, rgb, result.background_lab).items()
        }
    json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
