"""Read-only local A/B quality audit for Retake exports.

Measures residual face-outline darkness and photographic microdetail; does not
identify people, upload photos, or treat ArcFace cosine as photorealism.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .retake_rim_cleanup import inspect_dark_rim_file
from .ultra_quality import check_photo_detail


def audit_result(
    target: Path, result: Path, bounds: dict[str, float]
) -> dict[str, object]:
    report = inspect_dark_rim_file(target, result, bounds)
    detail, regions = check_photo_detail(target, result, bounds)
    valid_rim = report.get("stage") != "not_examined"
    fraction = report.get("dark_line_fraction") if valid_rim else None
    if isinstance(fraction, (int, float)) and math.isfinite(fraction):
        fraction = round(float(fraction), 5)
    else:
        fraction = None
    return {
        "residual_dark_seam_fraction": fraction,
        "face_detail_score": detail,
        "face_detail_ratios": regions,
        "dark_line_pixels": int(report.get("dark_line_pixels", 0)),
        "diagnostic_stage": str(report.get("stage", "not_examined")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retake exported photo seam/detail diagnostics (no face identity measurement)."
    )
    parser.add_argument("original", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument(
        "--bounds", nargs=4, type=float, metavar=("X", "Y", "WIDTH", "HEIGHT"),
        required=True, help="Face box in original-resolution pixels."
    )
    parser.add_argument("--previous", type=Path, help="Optional older Retake output for A/B")
    args = parser.parse_args()
    x, y, width, height = args.bounds
    if not all(math.isfinite(item) for item in (x, y, width, height)) or min(width, height) < 40:
        parser.error("Invalid face bounds.")
    for candidate in (args.original, args.result, args.previous):
        if candidate is not None and not candidate.is_file():
            parser.error("Input image not found.")
    box = {"x": x, "y": y, "width": width, "height": height}
    current = audit_result(args.original, args.result, box)
    output: dict[str, object] = {
        "note": "Smaller seam fraction is generally preferable; not a calibrated percentage of realism or identity.",
        "current": current,
    }
    if args.previous:
        old = audit_result(args.original, args.previous, box)
        output["previous"] = old
        now, before = current["residual_dark_seam_fraction"], old["residual_dark_seam_fraction"]
        output["seam_fraction_change"] = round(float(now) - float(before), 5) if now is not None and before is not None else None
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
