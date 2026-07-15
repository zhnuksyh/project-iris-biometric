"""
scripts/recover_preprocessing.py

Recover additional CASIA-IrisV4 samples that were skipped by the production
preprocessing pass.

The baseline tensors in data/processed/ are never touched. This script scans
the existing raw subset roots, finds raw JPGs whose baseline .npy is missing,
then tries controlled segmentation fallbacks and writes successful tensors to
data/processed_recovered/ with the same relative layout.

Usage
-----
    python -m scripts.recover_preprocessing
    python -m scripts.recover_preprocessing --subset CASIA-Iris-Lamp --limit 100
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.batch_processor import SUBSETS
from src.preprocessing.segmentation import (
    DEFAULT_SEGMENTATION_CONFIG,
    denoise_image,
    draw_segmentation_overlay,
    normalize_iris,
    scale_pixels,
    segment_iris_configurable,
    validate_iris_circles,
)


BASELINE_ROOT = "data/processed"
RECOVERED_ROOT = "data/processed_recovered"
METADATA_PATH = "reports/recovery_metadata.jsonl"
SUMMARY_PATH = "reports/recovery_summary.json"
QC_DIR = "figures/recovery_qc"


RECOVERY_METHODS = [
    {
        "name": "standard_retry",
        "preprocess": "standard",
        "config": {},
    },
    {
        "name": "clahe_default",
        "preprocess": "clahe",
        "config": {},
    },
    {
        "name": "clahe_relaxed",
        "preprocess": "clahe",
        "config": {
            "pupil": {
                "param2_start": 45,
                "param2_min": 3,
                "min_radius": 8,
                "max_radius": 95,
            },
            "iris": {
                "param2_start": 28,
                "param2_min": 3,
                "min_radius": 65,
                "max_radius": 240,
            },
            "center_offset_frac": 0.80,
            "center_offset_floor": 90.0,
        },
    },
    {
        "name": "wide_relaxed",
        "preprocess": "standard",
        "config": {
            "pupil": {
                "param2_start": 45,
                "param2_min": 3,
                "min_radius": 8,
                "max_radius": 100,
            },
            "iris": {
                "param2_start": 25,
                "param2_min": 3,
                "min_radius": 55,
                "max_radius": 260,
            },
            "center_offset_frac": 0.90,
            "center_offset_floor": 110.0,
        },
    },
]


def _merge_config(config):
    merged = {
        "pupil": DEFAULT_SEGMENTATION_CONFIG["pupil"].copy(),
        "iris": DEFAULT_SEGMENTATION_CONFIG["iris"].copy(),
        "center_offset_frac": DEFAULT_SEGMENTATION_CONFIG["center_offset_frac"],
        "center_offset_floor": DEFAULT_SEGMENTATION_CONFIG["center_offset_floor"],
    }
    for key in ("pupil", "iris"):
        if key in config:
            merged[key].update(config[key])
    for key in ("center_offset_frac", "center_offset_floor"):
        if key in config:
            merged[key] = config[key]
    return merged


def _preprocess_image(image_path: str, mode: str) -> np.ndarray:
    if mode == "standard":
        return denoise_image(image_path)
    if mode == "clahe":
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        image = cv2.medianBlur(image, ksize=5)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        return cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=1.5)
    raise ValueError(f"Unknown preprocess mode: {mode}")


def _baseline_output_path(subset_name: str, subset_root: str,
                          image_path: str, root: str) -> str:
    rel_path = os.path.relpath(image_path, subset_root)
    rel_npy = os.path.splitext(rel_path)[0] + ".npy"
    return os.path.join(root, subset_name, rel_npy)


def _validate_tensor(tensor: np.ndarray) -> tuple:
    if tensor.shape != (128, 128, 1):
        return False, f"bad_shape:{tensor.shape}"
    if tensor.dtype != np.float32:
        return False, f"bad_dtype:{tensor.dtype}"
    if not np.isfinite(tensor).all():
        return False, "non_finite_tensor"
    if float(tensor.min()) < 0.0 or float(tensor.max()) > 1.0:
        return False, "tensor_out_of_range"
    return True, "ok"


def _jsonable_circles(circles: dict) -> dict:
    return {
        "center": [int(circles["center"][0]), int(circles["center"][1])],
        "r_pupil": float(circles["r_pupil"]),
        "r_iris": float(circles["r_iris"]),
        "pupil_center": list(circles.get("pupil_center", circles["center"])),
        "iris_center": list(circles.get("iris_center", circles["center"])),
        "center_distance": float(circles.get("center_distance", 0.0)),
    }


def _try_recover(image_path: str, subset_name: str, subset_root: str,
                 recovered_root: str, qc_dir: str, qc_index: int,
                 qc_limit: int) -> dict:
    out_path = _baseline_output_path(subset_name, subset_root, image_path,
                                     recovered_root)
    record = {
        "source_path": image_path,
        "subset": subset_name,
        "output_path": out_path,
        "status": "failed",
        "method": None,
        "failure_reason": "no_method_succeeded",
    }

    for method in RECOVERY_METHODS:
        try:
            image = _preprocess_image(image_path, method["preprocess"])
            config = _merge_config(method["config"])
            circles = segment_iris_configurable(image, config)
            if circles is None:
                record["failure_reason"] = f"{method['name']}:segmentation_failed"
                continue

            ok, reason = validate_iris_circles(circles, image.shape)
            if not ok:
                record["failure_reason"] = f"{method['name']}:{reason}"
                continue

            strip = normalize_iris(image, circles, circles)
            tensor = scale_pixels(strip)
            ok, reason = _validate_tensor(tensor)
            if not ok:
                record["failure_reason"] = f"{method['name']}:{reason}"
                continue

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, tensor)

            if qc_index < qc_limit:
                os.makedirs(qc_dir, exist_ok=True)
                overlay = draw_segmentation_overlay(image, circles)
                qc_name = f"{qc_index:04d}_{subset_name}_{Path(image_path).stem}.jpg"
                cv2.imwrite(os.path.join(qc_dir, qc_name), overlay)

            record.update({
                "status": "recovered",
                "method": method["name"],
                "failure_reason": None,
                "circles": _jsonable_circles(circles),
            })
            return record
        except Exception as exc:
            record["failure_reason"] = f"{method['name']}:error:{exc}"

    return record


def _iter_raw_images(subset_filter=None):
    for subset_name, subset_root in SUBSETS.items():
        if subset_filter and subset_name not in subset_filter:
            continue
        if not os.path.isdir(subset_root):
            yield {
                "status": "missing_subset",
                "subset": subset_name,
                "source_root": subset_root,
            }
            continue
        for dirpath, _, filenames in os.walk(subset_root):
            for filename in sorted(filenames):
                if filename.lower().endswith(".jpg"):
                    yield {
                        "status": "candidate",
                        "subset": subset_name,
                        "subset_root": subset_root,
                        "source_path": os.path.join(dirpath, filename),
                    }


def run(args):
    subset_filter = set(args.subset) if args.subset else None
    os.makedirs(os.path.dirname(args.metadata), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)

    counts = Counter()
    processed = 0
    qc_successes = 0

    metadata_mode = "w" if args.overwrite_metadata else "a"
    with open(args.metadata, metadata_mode) as meta:
        for item in _iter_raw_images(subset_filter):
            if item["status"] == "missing_subset":
                counts["missing_subset"] += 1
                meta.write(json.dumps(item) + "\n")
                continue

            subset = item["subset"]
            subset_root = item["subset_root"]
            src = item["source_path"]
            baseline_path = _baseline_output_path(
                subset, subset_root, src, args.baseline_root,
            )
            recovered_path = _baseline_output_path(
                subset, subset_root, src, args.recovered_root,
            )

            if os.path.isfile(baseline_path):
                counts["already_baseline"] += 1
                continue
            if os.path.isfile(recovered_path) and not args.overwrite_outputs:
                counts["already_recovered"] += 1
                continue

            record = _try_recover(
                src, subset, subset_root, args.recovered_root,
                args.qc_dir, qc_successes, args.qc_limit,
            )
            record["baseline_path"] = baseline_path
            meta.write(json.dumps(record) + "\n")
            counts[record["status"]] += 1
            if record["status"] == "recovered":
                counts[f"recovered:{record['method']}"] += 1
                qc_successes += 1
            else:
                counts[record["failure_reason"]] += 1

            processed += 1
            if args.limit and processed >= args.limit:
                break
            if processed % args.progress_every == 0:
                print(f"[recover] tried={processed} recovered={counts['recovered']} "
                      f"failed={counts['failed']}")

    summary = {
        "baseline_root": args.baseline_root,
        "recovered_root": args.recovered_root,
        "metadata_path": args.metadata,
        "qc_dir": args.qc_dir,
        "counts": dict(counts),
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", default=BASELINE_ROOT)
    parser.add_argument("--recovered-root", default=RECOVERED_ROOT)
    parser.add_argument("--metadata", default=METADATA_PATH)
    parser.add_argument("--summary", default=SUMMARY_PATH)
    parser.add_argument("--qc-dir", default=QC_DIR)
    parser.add_argument("--qc-limit", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum skipped images to try; 0 means no limit")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--subset", action="append",
                        help="Subset name to process; may be passed multiple times")
    parser.add_argument("--overwrite-metadata", action="store_true")
    parser.add_argument("--overwrite-outputs", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
