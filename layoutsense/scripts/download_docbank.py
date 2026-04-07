#!/usr/bin/env python3
"""
DocBank Dataset Downloader & Preprocessor
==========================================
DocBank (Li et al., 2020) provides 500K document pages with token-level
bounding box annotations and semantic labels. We use it to extract
spatial features for training the layout classifier.

Labels in DocBank:
    abstract, author, caption, date, equation, figure, footer,
    list, paragraph, reference, section, table, title, header

We map these to DocuParse's RegionLabel taxonomy before feature extraction.

Usage:
    python scripts/download_docbank.py --output data/docbank --sample 5000
    python scripts/download_docbank.py --output data/docbank --full

Dataset source: https://github.com/doc-analysis/DocBank
Hugging Face mirror: datasets library (doc-analysis/docbank)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── DocBank label → DocuParse RegionLabel mapping ────────────────────────────
DOCBANK_TO_DOCUPARSE = {
    "title":     "heading",
    "section":   "heading",
    "abstract":  "paragraph",
    "paragraph": "paragraph",
    "list":      "paragraph",
    "reference": "paragraph",
    "author":    "header",
    "date":      "header",
    "header":    "header",
    "footer":    "footer",
    "caption":   "caption",
    "figure":    "figure",
    "table":     "table",
    "equation":  "paragraph",  # treat as paragraph for our taxonomy
}

# Features we extract per bounding-box region
FEATURE_NAMES = [
    "x_min_norm",        # x_min / page_width
    "y_min_norm",        # y_min / page_height
    "x_max_norm",
    "y_max_norm",
    "width_norm",        # width / page_width
    "height_norm",       # height / page_height
    "aspect_ratio",      # width / height
    "area_norm",         # area / page_area
    "x_center_norm",
    "y_center_norm",
    "height_vs_median",  # box height / page median token height
    "x_start_bucket",    # quantised x_min (0–9)
    "y_position_frac",   # y_center / page_height (0=top, 1=bottom)
    "width_frac_page",   # how much of page width the box spans
    "is_wide",           # width_norm > 0.5
    "is_top_margin",     # y_center < 0.08
    "is_bottom_margin",  # y_center > 0.92
    "n_tokens_in_region",# number of tokens grouped into this region
    "avg_token_height",  # mean token height in region (proxy for font size)
    "token_height_std",  # std of token heights (uniformity signal)
]


def _try_import_datasets():
    try:
        from datasets import load_dataset
        return load_dataset
    except ImportError:
        logger.error("Install with: pip install datasets")
        sys.exit(1)


def extract_features_from_docbank_sample(
    sample: dict,
    page_width: int = 1000,
    page_height: int = 1000,
) -> Optional[Tuple[np.ndarray, str]]:
    """
    Extract a feature vector and label from one DocBank token annotation.

    DocBank provides per-token bounding boxes. We group tokens by their
    semantic label into regions (same label, spatially proximate) and
    extract region-level features.

    Returns (feature_vector, docuparse_label) or None if label unknown.
    """
    label = sample.get("label", "")
    docuparse_label = DOCBANK_TO_DOCUPARSE.get(label)
    if docuparse_label is None:
        return None

    # DocBank bounding box: [x0, y0, x1, y1] normalised to 1000×1000
    bbox = sample.get("bounding_box", [])
    if not bbox or len(bbox) < 4:
        return None

    x_min, y_min, x_max, y_max = [float(v) for v in bbox[:4]]

    # Normalise to [0, 1]
    xn = lambda v: v / page_width
    yn = lambda v: v / page_height

    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        return None

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    area = width * height

    features = [
        xn(x_min),
        yn(y_min),
        xn(x_max),
        yn(y_max),
        xn(width),
        yn(height),
        width / height,
        area / (page_width * page_height),
        xn(x_center),
        yn(y_center),
        yn(height) / 0.02,        # height vs typical token height (~20/1000)
        int(xn(x_min) * 10),      # x bucket 0–9
        yn(y_center),
        xn(width),
        float(xn(width) > 0.5),
        float(yn(y_center) < 0.08),
        float(yn(y_center) > 0.92),
        1.0,                      # single token; aggregated later
        yn(height),
        0.0,                      # std undefined for single token
    ]

    return np.array(features, dtype=np.float32), docuparse_label


def load_and_preprocess(
    output_dir: Path,
    n_samples: Optional[int] = 5000,
    split: str = "train",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Download DocBank via HuggingFace datasets and build feature matrix.

    Returns
    -------
    X : np.ndarray, shape (n, len(FEATURE_NAMES))
    y : np.ndarray, shape (n,) — integer class labels
    class_names : List[str] — index → label name
    """
    load_dataset = _try_import_datasets()

    logger.info("Loading DocBank from HuggingFace Hub (split=%s, samples=%s)...", split, n_samples or "all")

    # DocBank is large (~500K items); streaming avoids downloading everything
    dataset = load_dataset(
        "doc-analysis/DocBank",
        split=split,
        streaming=(n_samples is not None),
        trust_remote_code=True,
    )

    if n_samples:
        dataset = dataset.take(n_samples * 3)  # Oversample to account for filtering

    features_list: List[np.ndarray] = []
    labels_list: List[str] = []
    skipped = 0

    for sample in dataset:
        result = extract_features_from_docbank_sample(sample)
        if result is None:
            skipped += 1
            continue
        feat, label = result
        features_list.append(feat)
        labels_list.append(label)
        if n_samples and len(features_list) >= n_samples:
            break

    logger.info("Extracted %d samples (%d skipped/unknown label)", len(features_list), skipped)

    label_counts = Counter(labels_list)
    logger.info("Class distribution: %s", dict(label_counts.most_common()))

    # Encode labels
    class_names = sorted(set(labels_list))
    label_to_idx = {l: i for i, l in enumerate(class_names)}
    y = np.array([label_to_idx[l] for l in labels_list], dtype=np.int32)
    X = np.vstack(features_list)

    # Save to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_train.npy", X)
    np.save(output_dir / "y_train.npy", y)
    (output_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))
    (output_dir / "feature_names.json").write_text(json.dumps(FEATURE_NAMES, indent=2))

    logger.info("Saved to %s", output_dir)
    return X, y, class_names


def load_publanet_for_eval(
    output_dir: Path,
    n_samples: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load PubLayNet for benchmark evaluation (separate from training data).
    PubLayNet labels: text, title, list, table, figure
    """
    PUBLANET_TO_DOCUPARSE = {
        "text":   "paragraph",
        "title":  "heading",
        "list":   "paragraph",
        "table":  "table",
        "figure": "figure",
    }

    load_dataset = _try_import_datasets()
    logger.info("Loading PubLayNet for evaluation...")

    try:
        dataset = load_dataset(
            "ds4sd/PubLayNet",
            split="validation",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.warning("PubLayNet HF load failed (%s) — using DocBank test split", e)
        return _load_docbank_test_split(output_dir, n_samples)

    features_list, labels_list = [], []

    for sample in dataset:
        # PubLayNet has region-level annotations with bbox and category
        for annotation in sample.get("annotations", []):
            cat = annotation.get("category_name", "")
            label = PUBLANET_TO_DOCUPARSE.get(cat)
            if label is None:
                continue

            bbox = annotation.get("bbox", [])  # [x, y, w, h] COCO format
            if len(bbox) < 4:
                continue

            x_min, y_min, w, h = bbox
            x_max, y_max = x_min + w, y_min + h
            img_w = sample.get("width", 1000)
            img_h = sample.get("height", 1000)

            if w <= 0 or h <= 0:
                continue

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            features = [
                x_min / img_w, y_min / img_h,
                x_max / img_w, y_max / img_h,
                w / img_w, h / img_h,
                w / h,
                (w * h) / (img_w * img_h),
                x_center / img_w,
                y_center / img_h,
                (h / img_h) / 0.02,
                int((x_min / img_w) * 10),
                y_center / img_h,
                w / img_w,
                float(w / img_w > 0.5),
                float(y_center / img_h < 0.08),
                float(y_center / img_h > 0.92),
                1.0, h / img_h, 0.0,
            ]

            features_list.append(np.array(features, dtype=np.float32))
            labels_list.append(label)

            if len(features_list) >= n_samples:
                break
        if len(features_list) >= n_samples:
            break

    if not features_list:
        return _load_docbank_test_split(output_dir, n_samples)

    class_names = sorted(set(labels_list))
    label_to_idx = {l: i for i, l in enumerate(class_names)}
    y = np.array([label_to_idx[l] for l in labels_list], dtype=np.int32)
    X = np.vstack(features_list)

    np.save(output_dir / "X_eval.npy", X)
    np.save(output_dir / "y_eval.npy", y)
    (output_dir / "eval_class_names.json").write_text(json.dumps(class_names))

    logger.info("PubLayNet eval: %d samples, classes=%s", len(features_list), class_names)
    return X, y, class_names


def _load_docbank_test_split(output_dir: Path, n_samples: int):
    """Fallback: use DocBank test split for evaluation."""
    load_dataset = _try_import_datasets()
    dataset = load_dataset("doc-analysis/DocBank", split="test", streaming=True, trust_remote_code=True)
    features_list, labels_list = [], []
    for sample in dataset:
        result = extract_features_from_docbank_sample(sample)
        if result:
            features_list.append(result[0])
            labels_list.append(result[1])
        if len(features_list) >= n_samples:
            break
    class_names = sorted(set(labels_list))
    label_to_idx = {l: i for i, l in enumerate(class_names)}
    y = np.array([label_to_idx[l] for l in labels_list], dtype=np.int32)
    X = np.vstack(features_list)
    np.save(output_dir / "X_eval.npy", X)
    np.save(output_dir / "y_eval.npy", y)
    (output_dir / "eval_class_names.json").write_text(json.dumps(class_names))
    return X, y, class_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess DocBank")
    parser.add_argument("--output", type=Path, default=Path("data/docbank"))
    parser.add_argument("--sample", type=int, default=5000, help="Training samples (0=full dataset)")
    parser.add_argument("--full", action="store_true", help="Download full dataset (slow)")
    parser.add_argument("--eval", action="store_true", help="Also download PubLayNet eval set")
    args = parser.parse_args()

    n = None if args.full else args.sample
    X, y, classes = load_and_preprocess(args.output, n_samples=n)
    print(f"\nReady: X={X.shape}, y={y.shape}, classes={classes}")

    if args.eval:
        X_e, y_e, c_e = load_publanet_for_eval(args.output)
        print(f"Eval:  X={X_e.shape}, y={y_e.shape}, classes={c_e}")
