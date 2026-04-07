"""
Trained Layout Classifier
=========================
Replaces the hardcoded threshold heuristics in layout_analyser.py with
a LightGBM model trained on DocBank spatial features.

The key distinction from the original heuristic approach:
- HEADING_HEIGHT_RATIO=1.4 was an arbitrary constant.
- This classifier learned the actual decision boundary from ~500K ground-truth
  DocBank annotations, so the threshold for "this is a heading" is data-driven.

Feature extraction mirrors the DocBank preprocessing pipeline exactly so that
the feature space at inference matches what the model was trained on.

This module is designed to be a drop-in replacement for _classify_region()
in layout_analyser.py — same interface, trained weights instead of rules.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from app.models.schemas import RegionLabel, TextBlock

logger = logging.getLogger(__name__)

# Path to the trained model artifact — populated by train_classifier.py
_DEFAULT_MODEL_PATH = Path(__file__).parent / "model" / "layout_classifier.lgb"
_DEFAULT_META_PATH = Path(__file__).parent / "model" / "model_meta.json"

_model_lock = threading.Lock()
_classifier_instance: Optional["LayoutClassifier"] = None


class LayoutClassifier:
    """
    LightGBM-based layout region classifier.

    Encapsulates model loading, feature extraction, and inference.
    Exposes a single `predict()` method that mirrors the interface
    of the heuristic _classify_region() function it replaces.
    """

    def __init__(self, model_path: Path = _DEFAULT_MODEL_PATH):
        self._model = None
        self._class_names: List[str] = []
        self._label_to_enum: dict = {}
        self._model_path = model_path
        self._load(model_path)

    def _load(self, model_path: Path):
        """Load the LightGBM model and metadata from disk."""
        meta_path = model_path.parent / "model_meta.json"

        if not model_path.exists():
            logger.warning(
                "Trained model not found at %s. "
                "Falling back to heuristic classifier. "
                "Run: python scripts/train_classifier.py",
                model_path,
            )
            return

        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("lightgbm not installed — falling back to heuristics. pip install lightgbm")
            return

        try:
            self._model = lgb.Booster(model_file=str(model_path))
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                self._class_names = meta["class_names"]
            else:
                # Infer class count from model
                self._class_names = [f"class_{i}" for i in range(self._model.num_class())]

            # Map string label → RegionLabel enum
            self._label_to_enum = {
                "heading":   RegionLabel.HEADING,
                "paragraph": RegionLabel.PARAGRAPH,
                "table":     RegionLabel.TABLE,
                "figure":    RegionLabel.FIGURE,
                "caption":   RegionLabel.CAPTION,
                "header":    RegionLabel.HEADER,
                "footer":    RegionLabel.FOOTER,
            }

            logger.info(
                "Layout classifier loaded from %s (%d classes: %s)",
                model_path, len(self._class_names), self._class_names,
            )
        except Exception as exc:
            logger.error("Failed to load classifier: %s — using heuristics", exc)
            self._model = None

    @property
    def is_available(self) -> bool:
        return self._model is not None

    # ─────────────────────────────────────────────────────────────────────────
    # Feature extraction (must match download_docbank.py exactly)
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_features(
        self,
        blocks: List[TextBlock],
        page_width: int,
        page_height: int,
        median_token_height: float,
    ) -> np.ndarray:
        """
        Build a 20-dimensional feature vector for a cluster of TextBlocks.

        This is the inference-time mirror of extract_features_from_docbank_sample()
        in download_docbank.py. The features must be identical — any mismatch
        between training and inference features causes silent accuracy degradation.
        """
        if not blocks:
            return np.zeros(20, dtype=np.float32)

        # Aggregate bounding box across all blocks in the region
        x_min = min(b.bounding_box.x_min for b in blocks)
        y_min = min(b.bounding_box.y_min for b in blocks)
        x_max = max(b.bounding_box.x_max for b in blocks)
        y_max = max(b.bounding_box.y_max for b in blocks)

        w = x_max - x_min
        h = y_max - y_min

        if w <= 0 or h <= 0:
            return np.zeros(20, dtype=np.float32)

        pw, ph = float(page_width), float(page_height)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        area = w * h

        # Per-token heights for std calculation
        token_heights = [b.bounding_box.height for b in blocks]
        avg_token_h = float(np.mean(token_heights))
        std_token_h = float(np.std(token_heights)) if len(token_heights) > 1 else 0.0

        # Normalise token height vs page scale (same as DocBank's 1000-unit space)
        # DocBank uses absolute pixel coords in a 1000×1000 normalised space
        # We replicate by using page-relative coordinates
        scale = 1000.0
        h_norm_docbank = (h / ph) * scale  # height in DocBank coordinate space proxy

        features = np.array([
            x_min / pw,                          # x_min_norm
            y_min / ph,                          # y_min_norm
            x_max / pw,                          # x_max_norm
            y_max / ph,                          # y_max_norm
            w / pw,                              # width_norm
            h / ph,                              # height_norm
            w / h,                               # aspect_ratio
            area / (pw * ph),                    # area_norm
            x_center / pw,                       # x_center_norm
            y_center / ph,                       # y_center_norm
            (h / ph) / 0.02,                     # height_vs_median (0.02 = typical token)
            float(int((x_min / pw) * 10)),       # x_start_bucket
            y_center / ph,                       # y_position_frac
            w / pw,                              # width_frac_page
            float(w / pw > 0.5),                 # is_wide
            float(y_center / ph < 0.08),         # is_top_margin
            float(y_center / ph > 0.92),         # is_bottom_margin
            float(len(blocks)),                  # n_tokens_in_region
            avg_token_h / ph,                    # avg_token_height (normalised)
            std_token_h / ph,                    # token_height_std (normalised)
        ], dtype=np.float32)

        return features

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        blocks: List[TextBlock],
        page_width: int,
        page_height: int,
        median_token_height: float,
    ) -> Tuple[RegionLabel, float]:
        """
        Predict the semantic label for a cluster of TextBlocks.

        Parameters
        ----------
        blocks : List[TextBlock]
            All OCR TextBlocks belonging to this region cluster.
        page_width, page_height : int
            Page dimensions in pixels.
        median_token_height : float
            Median bounding box height across all blocks on the page.

        Returns
        -------
        (RegionLabel, confidence) — same interface as heuristic _classify_region()
        """
        if not self.is_available:
            # Graceful fallback to heuristics if model not loaded
            from app.services.layout_analyser import _classify_region, _median_char_height
            return _classify_region(blocks, page_width, page_height, median_token_height)

        features = self._extract_features(blocks, page_width, page_height, median_token_height)
        features_2d = features.reshape(1, -1)

        # LightGBM returns a probability distribution over classes
        proba = self._model.predict(features_2d)[0]
        best_idx = int(np.argmax(proba))
        confidence = float(proba[best_idx])

        predicted_label_str = self._class_names[best_idx] if best_idx < len(self._class_names) else "paragraph"
        label = self._label_to_enum.get(predicted_label_str, RegionLabel.UNKNOWN)

        return label, round(confidence, 4)

    def predict_batch(
        self,
        feature_matrix: np.ndarray,
    ) -> Tuple[List[str], List[float]]:
        """
        Batch prediction on a pre-built feature matrix.
        Used by the benchmark script for efficient eval on large datasets.
        """
        if not self.is_available:
            raise RuntimeError("Model not loaded")

        probas = self._model.predict(feature_matrix)
        best_idxs = np.argmax(probas, axis=1)
        labels = [self._class_names[i] if i < len(self._class_names) else "paragraph" for i in best_idxs]
        confidences = [float(probas[i, best_idxs[i]]) for i in range(len(probas))]
        return labels, confidences


def get_classifier() -> LayoutClassifier:
    """Singleton accessor — initialise once, reuse across requests."""
    global _classifier_instance
    if _classifier_instance is not None:
        return _classifier_instance
    with _model_lock:
        if _classifier_instance is None:
            _classifier_instance = LayoutClassifier()
    return _classifier_instance
