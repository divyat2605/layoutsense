#!/usr/bin/env python3
"""
PubLayNet Benchmark Harness
============================
Evaluates DocuParse's layout classifier against PubLayNet's validation split
and reports precision, recall, and F1 per class.

This is the evaluation artifact that transforms the project from
"I built a system" to "I measured a system."

Outputs:
  benchmark/publanet_results.json   — machine-readable metrics
  benchmark/publanet_report.md      — human-readable report for README

Usage:
    # Quick eval (2000 samples)
    python benchmark/run_benchmark.py --samples 2000

    # Full eval
    python benchmark/run_benchmark.py --full

    # Also benchmark LayoutLMv3 re-scoring (slow, needs GPU for reasonable speed)
    python benchmark/run_benchmark.py --samples 2000 --include-layoutlmv3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# PubLayNet label → DocuParse RegionLabel
PUBLANET_LABEL_MAP = {
    "text":   "paragraph",
    "title":  "heading",
    "list":   "paragraph",
    "table":  "table",
    "figure": "figure",
}

ALL_CLASSES = ["heading", "paragraph", "table", "figure", "caption", "header", "footer"]


def load_publanet_streaming(n_samples: Optional[int] = 2000):
    """
    Stream PubLayNet validation annotations and build (features, labels) pairs.
    Falls back to DocBank test split if PubLayNet is unavailable.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install datasets")
        sys.exit(1)

    from scripts.download_docbank import extract_features_from_docbank_sample

    logger.info("Loading PubLayNet validation split (streaming)...")
    try:
        dataset = load_dataset("ds4sd/PubLayNet", split="validation", streaming=True, trust_remote_code=True)
        source = "PubLayNet"
    except Exception as e:
        logger.warning("PubLayNet unavailable (%s) — using DocBank test split", e)
        dataset = load_dataset("doc-analysis/DocBank", split="test", streaming=True, trust_remote_code=True)
        source = "DocBank (test split)"

    features, labels = [], []

    for sample in dataset:
        # PubLayNet format: sample has 'annotations' list with bbox + category_name
        if "annotations" in sample:
            for ann in sample.get("annotations", []):
                cat = ann.get("category_name", "")
                label = PUBLANET_LABEL_MAP.get(cat)
                if label is None:
                    continue
                bbox = ann.get("bbox", [])
                if len(bbox) < 4:
                    continue
                x, y, w, h = bbox
                iw = sample.get("width", 1000)
                ih = sample.get("height", 1000)
                if w <= 0 or h <= 0:
                    continue
                xc, yc = (x + w/2), (y + h/2)
                feat = np.array([
                    x/iw, y/ih, (x+w)/iw, (y+h)/ih,
                    w/iw, h/ih, w/h, (w*h)/(iw*ih),
                    xc/iw, yc/ih, (h/ih)/0.02, float(int(x/iw*10)),
                    yc/ih, w/iw, float(w/iw > 0.5),
                    float(yc/ih < 0.08), float(yc/ih > 0.92),
                    1.0, h/ih, 0.0,
                ], dtype=np.float32)
                features.append(feat)
                labels.append(label)
        else:
            # DocBank format: single annotation per sample
            result = extract_features_from_docbank_sample(sample)
            if result:
                features.append(result[0])
                labels.append(result[1])

        if n_samples and len(features) >= n_samples:
            break

    logger.info("Loaded %d eval samples from %s", len(features), source)
    return np.vstack(features), labels, source


def evaluate_lightgbm(
    X: np.ndarray,
    y_true: List[str],
    model_path: Path,
) -> Dict:
    """Run LightGBM classifier and return per-class metrics."""
    try:
        import lightgbm as lgb
        from sklearn.metrics import classification_report, precision_recall_fscore_support
    except ImportError:
        logger.error("pip install lightgbm scikit-learn")
        sys.exit(1)

    meta_path = model_path.parent / "model_meta.json"
    if not model_path.exists():
        logger.error("Model not found at %s — run scripts/train_classifier.py first", model_path)
        return {}

    model = lgb.Booster(model_file=str(model_path))
    class_names = json.loads(meta_path.read_text())["class_names"] if meta_path.exists() else ALL_CLASSES

    t0 = time.time()
    probas = model.predict(X)
    elapsed = time.time() - t0
    throughput = len(X) / elapsed

    pred_ids = np.argmax(probas, axis=1)
    pred_labels = [class_names[i] if i < len(class_names) else "paragraph" for i in pred_ids]

    # Align label spaces
    all_labels = sorted(set(y_true) | set(pred_labels))

    from sklearn.metrics import precision_recall_fscore_support, f1_score
    p, r, f, s = precision_recall_fscore_support(
        y_true, pred_labels, labels=all_labels, zero_division=0
    )

    return {
        "model": "LightGBM (DocBank-trained)",
        "n_samples": len(X),
        "inference_throughput_per_sec": round(throughput, 1),
        "macro_f1": round(float(f1_score(y_true, pred_labels, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, pred_labels, average="weighted", zero_division=0)), 4),
        "per_class": {
            label: {
                "precision": round(float(p[i]), 4),
                "recall": round(float(r[i]), 4),
                "f1": round(float(f[i]), 4),
                "support": int(s[i]),
            }
            for i, label in enumerate(all_labels)
        },
    }


def evaluate_heuristic(X: np.ndarray, y_true: List[str]) -> Dict:
    """
    Evaluate the original heuristic classifier as a baseline.
    Reconstructs TextBlock objects from feature vectors to drive the heuristic.
    """
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    from app.models.schemas import BoundingBox, TextBlock
    from app.services.layout_analyser import _classify_region, _median_char_height

    logger.info("Running heuristic baseline evaluation...")
    pred_labels = []

    for feat in X:
        # Reconstruct a minimal TextBlock from the feature vector
        # Features: x_min_norm, y_min_norm, x_max_norm, y_max_norm, ...
        page_w, page_h = 1000, 1000
        bb = BoundingBox(
            x_min=float(feat[0] * page_w),
            y_min=float(feat[1] * page_h),
            x_max=float(feat[2] * page_w),
            y_max=float(feat[3] * page_h),
        )
        block = TextBlock(text="word", confidence=0.9, bounding_box=bb, page_number=1)
        label, _ = _classify_region([block], page_w, page_h, median_height=12.0)
        pred_labels.append(label.value)

    all_labels = sorted(set(y_true) | set(pred_labels))
    p, r, f, s = precision_recall_fscore_support(
        y_true, pred_labels, labels=all_labels, zero_division=0
    )

    return {
        "model": "Heuristic (HEADING_HEIGHT_RATIO thresholds)",
        "n_samples": len(X),
        "macro_f1": round(float(f1_score(y_true, pred_labels, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, pred_labels, average="weighted", zero_division=0)), 4),
        "per_class": {
            label: {
                "precision": round(float(p[i]), 4),
                "recall": round(float(r[i]), 4),
                "f1": round(float(f[i]), 4),
                "support": int(s[i]),
            }
            for i, label in enumerate(all_labels)
        },
    }


def generate_markdown_report(
    results: List[Dict],
    source: str,
    output_path: Path,
):
    """Generate a markdown benchmark report suitable for inclusion in the README."""
    lines = [
        "# DocuParse Layout Classifier — Benchmark Results",
        "",
        f"**Evaluation dataset:** {source}",
        f"**Date:** {__import__('datetime').date.today()}",
        "",
        "## Summary",
        "",
        "| Model | Macro F1 | Weighted F1 | Throughput |",
        "|-------|----------|-------------|------------|",
    ]

    for r in results:
        throughput = f"{r.get('inference_throughput_per_sec', 'N/A')} regions/s"
        lines.append(
            f"| {r['model']} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} | {throughput} |"
        )

    lines += ["", "## Per-Class Breakdown", ""]

    for r in results:
        lines += [f"### {r['model']}", ""]
        lines += ["| Class | Precision | Recall | F1 | Support |"]
        lines += ["|-------|-----------|--------|----|---------|"]
        for cls, m in r["per_class"].items():
            if m["support"] > 0:
                lines.append(
                    f"| {cls:<12} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |"
                )
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- LightGBM model trained on DocBank (Li et al., 2020) spatial features.",
        "- Heuristic baseline uses fixed `HEADING_HEIGHT_RATIO=1.4` threshold.",
        "- Evaluation on PubLayNet validation split (Zhong et al., 2019).",
        "- Throughput measured on CPU (no GPU).",
    ]

    output_path.write_text("\n".join(lines))
    logger.info("Report written to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Run PubLayNet benchmark")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--full", action="store_true", help="Evaluate on all samples")
    parser.add_argument("--model", type=Path, default=Path("app/classifier/model/layout_classifier.lgb"))
    parser.add_argument("--output", type=Path, default=Path("benchmark"))
    parser.add_argument("--include-heuristic", action="store_true", help="Also benchmark heuristic baseline")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    n = None if args.full else args.samples

    # Load eval data
    X, y_true, source = load_publanet_streaming(n)

    all_results = []

    # LightGBM evaluation
    lgbm_results = evaluate_lightgbm(X, y_true, args.model)
    if lgbm_results:
        all_results.append(lgbm_results)
        (args.output / "publanet_results.json").write_text(json.dumps(lgbm_results, indent=2))
        logger.info("\n=== LightGBM Results ===")
        logger.info("Macro F1:    %.4f", lgbm_results["macro_f1"])
        logger.info("Weighted F1: %.4f", lgbm_results["weighted_f1"])
        for cls, m in lgbm_results["per_class"].items():
            if m["support"] > 0:
                logger.info("  %-12s  P=%.3f R=%.3f F1=%.3f (n=%d)",
                           cls, m["precision"], m["recall"], m["f1"], m["support"])

    # Heuristic baseline
    if args.include_heuristic:
        heuristic_results = evaluate_heuristic(X, y_true)
        all_results.append(heuristic_results)
        logger.info("\n=== Heuristic Baseline ===")
        logger.info("Macro F1: %.4f", heuristic_results["macro_f1"])

    # Generate markdown report
    if all_results:
        generate_markdown_report(all_results, source, args.output / "publanet_report.md")
        print(f"\nResults saved to {args.output}/")
        print(f"Macro F1: {all_results[0]['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
