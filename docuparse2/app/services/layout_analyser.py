"""
Layout Analysis — Spatial Clustering and Region Classification
=============================================================
Inspired by LayoutLM (Xu et al., 2020): "LayoutLM: Pre-training of Text
and Layout for Document Image Understanding"

Core insight from LayoutLM: bounding box coordinates are first-class
features for understanding document structure. A word at (x=50, y=80)
with a large bounding box height is almost certainly a heading, not a
body paragraph, regardless of its textual content.

Classification strategy (two-tier):
    Primary:  LightGBM classifier trained on DocBank ground-truth annotations
              (app/classifier/classifier.py). Learns decision boundaries from
              data rather than hand-tuned constants.
    Fallback: Heuristic rules (original implementation) used when the trained
              model is not available (model file missing or lightgbm not installed).

Full pipeline:
    1. Feature engineering: derive spatial and typographic features
       from raw TextBlock bounding boxes.
    2. DBSCAN clustering: group spatially proximate TextBlocks into
       candidate regions. DBSCAN is chosen over k-means because:
       (a) the number of regions is unknown a priori, and
       (b) it handles noise/outlier blocks naturally.
    3. Trained classification: LightGBM predicts RegionLabel from 20
       spatial features per cluster, with heuristic fallback.
    4. Reading order reconstruction: sort regions top-to-bottom,
       left-to-right (with column detection for multi-column layouts).
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from app.core.config import settings
from app.core.exceptions import LayoutAnalysisError
from app.models.schemas import (
    BoundingBox,
    LayoutRegion,
    PageResult,
    RegionLabel,
    TextBlock,
)
from app.services.ocr_pipeline import RawOCROutput

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def _extract_spatial_features(blocks: List[TextBlock]) -> np.ndarray:
    """
    Build a feature matrix for DBSCAN clustering.

    We use (y_center, x_center) as the primary clustering dimensions,
    weighted so that vertical proximity (same line) dominates.
    This mimics reading order: blocks on the same line cluster together
    before blocks that are merely horizontally nearby but vertically distant.

    Feature vector per block: [y_center_scaled, x_center_scaled]
    """
    features = []
    for block in blocks:
        bb = block.bounding_box
        y_center = (bb.y_min + bb.y_max) / 2.0
        x_center = (bb.x_min + bb.x_max) / 2.0
        features.append([y_center, x_center])
    return np.array(features, dtype=np.float64)


def _estimate_char_height(block: TextBlock) -> float:
    """
    Estimate the character height of a text block as a proxy for font size.
    The bounding box height divided by a heuristic line-height factor (1.2)
    approximates the cap height of the rendered text.
    """
    return block.bounding_box.height / 1.2


def _median_char_height(blocks: List[TextBlock]) -> float:
    """Return the median character height across all blocks on a page."""
    if not blocks:
        return 12.0
    heights = [_estimate_char_height(b) for b in blocks]
    return float(np.median(heights))


# ─────────────────────────────────────────────────────────────────────────────
# DBSCAN Clustering
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_blocks(blocks: List[TextBlock]) -> np.ndarray:
    """
    Run DBSCAN on spatial features to group TextBlocks into candidate regions.

    DBSCAN parameters:
    - eps: neighbourhood radius in feature space. We use separate eps values
      for Y and X by pre-scaling features — Y is divided by DBSCAN_EPS_Y
      and X by DBSCAN_EPS_X, so DBSCAN's single eps=1.0 corresponds to
      "within one eps-unit in each direction."
    - min_samples=1 allows every block to form its own cluster if isolated,
      which is appropriate because we don't want to discard any OCR output.

    Returns label array: -1 = noise (isolated block treated as own region).
    """
    if len(blocks) == 0:
        return np.array([], dtype=int)

    features = _extract_spatial_features(blocks)

    # Scale features so that eps=1.0 in DBSCAN corresponds to our thresholds
    scaled = np.column_stack([
        features[:, 0] / settings.DBSCAN_EPS_Y,   # Y axis
        features[:, 1] / settings.DBSCAN_EPS_X,   # X axis
    ])

    labels = DBSCAN(
        eps=1.0,
        min_samples=settings.DBSCAN_MIN_SAMPLES,
        metric="chebyshev",  # L-infinity norm handles non-isotropic clusters well
    ).fit_predict(scaled)

    # Reclassify noise points (-1) as singleton clusters
    next_label = int(labels.max()) + 1 if len(labels) > 0 else 0
    for i, lbl in enumerate(labels):
        if lbl == -1:
            labels[i] = next_label
            next_label += 1

    n_clusters = len(set(labels))
    logger.debug("DBSCAN: %d blocks → %d clusters", len(blocks), n_clusters)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Region Classification Heuristics
# ─────────────────────────────────────────────────────────────────────────────

def _enclosing_bbox(blocks: List[TextBlock]) -> BoundingBox:
    """Compute the axis-aligned bounding box enclosing all given blocks."""
    return BoundingBox(
        x_min=min(b.bounding_box.x_min for b in blocks),
        y_min=min(b.bounding_box.y_min for b in blocks),
        x_max=max(b.bounding_box.x_max for b in blocks),
        y_max=max(b.bounding_box.y_max for b in blocks),
    )


def _classify_region(
    blocks: List[TextBlock],
    page_width: int,
    page_height: int,
    median_height: float,
) -> Tuple[RegionLabel, float]:
    """
    Assign a semantic label to a cluster of TextBlocks.

    Heuristics derived from LayoutLM's observation that spatial position
    and text-box geometry are highly predictive of element type:

    HEADER / FOOTER:  Region centred in the top/bottom 8% of the page.
    HEADING:          Average char height significantly exceeds page median
                      OR the cluster has very few words (short, large text).
    TABLE:            Multiple blocks with strongly aligned x_min values
                      across different y-levels suggest a grid structure.
    FIGURE:           A single large, wide block with no recognisable text
                      (low avg confidence or very short text).
    CAPTION:          A short text block immediately below a figure region.
    PARAGRAPH:        Default — multi-line, normal-height text.
    """
    if not blocks:
        return RegionLabel.UNKNOWN, 0.5

    bbox = _enclosing_bbox(blocks)
    avg_height = float(np.mean([_estimate_char_height(b) for b in blocks]))
    avg_conf = float(np.mean([b.confidence for b in blocks]))
    total_words = sum(len(b.text.split()) for b in blocks)

    # ── Page margin detection ─────────────────────────────────────────────────
    y_center = (bbox.y_min + bbox.y_max) / 2.0
    if y_center < page_height * 0.08:
        return RegionLabel.HEADER, 0.80
    if y_center > page_height * 0.92:
        return RegionLabel.FOOTER, 0.80

    # ── Heading detection ─────────────────────────────────────────────────────
    height_ratio = avg_height / median_height if median_height > 0 else 1.0
    is_large_text = height_ratio >= settings.HEADING_HEIGHT_RATIO
    is_short_cluster = total_words <= 8
    if is_large_text and is_short_cluster:
        confidence = min(0.95, 0.7 + (height_ratio - settings.HEADING_HEIGHT_RATIO) * 0.15)
        return RegionLabel.HEADING, round(confidence, 3)

    # ── Table detection ───────────────────────────────────────────────────────
    if len(blocks) >= 4:
        x_starts = sorted([b.bounding_box.x_min for b in blocks])
        if _has_column_alignment(blocks, tolerance=settings.TABLE_ALIGNMENT_TOLERANCE):
            return RegionLabel.TABLE, 0.75

    # ── Figure detection (low-text, large bounding box) ───────────────────────
    if (
        bbox.area > settings.FIGURE_MIN_AREA_PX
        and total_words <= 3
        and avg_conf < 0.6
    ):
        return RegionLabel.FIGURE, 0.65

    # ── Default: paragraph ────────────────────────────────────────────────────
    return RegionLabel.PARAGRAPH, 0.85


def _has_column_alignment(blocks: List[TextBlock], tolerance: float) -> bool:
    """
    Detect whether blocks exhibit grid-like column alignment,
    indicative of a table structure.

    Approach: cluster x_min values; if 3+ distinct x-columns exist and
    2+ y-rows each contain multiple x-column members, it's likely a table.
    """
    x_starts = np.array([b.bounding_box.x_min for b in blocks]).reshape(-1, 1)
    col_labels = DBSCAN(eps=tolerance, min_samples=1).fit_predict(x_starts)
    n_columns = len(set(col_labels))

    if n_columns < 2:
        return False

    # Check that multiple y-rows have members in different columns
    y_centers = np.array([(b.bounding_box.y_min + b.bounding_box.y_max) / 2 for b in blocks]).reshape(-1, 1)
    row_labels = DBSCAN(eps=tolerance * 2, min_samples=1).fit_predict(y_centers)
    n_rows = len(set(row_labels))

    return n_columns >= 2 and n_rows >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Reading Order Reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _detect_columns(regions: List[LayoutRegion], page_width: int) -> Dict[str, int]:
    """
    Assign a column index to each region for multi-column reading order.
    Uses x-center clustering: a two-column layout will show two x-center
    clusters separated by the page midpoint.
    """
    if not regions:
        return {}

    x_centers = np.array([
        [(r.bounding_box.x_min + r.bounding_box.x_max) / 2]
        for r in regions
    ])
    col_labels = DBSCAN(eps=page_width * 0.15, min_samples=1).fit_predict(x_centers)

    # Map cluster label → sorted column index (left to right)
    unique_labels = sorted(set(col_labels), key=lambda lbl: np.mean(
        x_centers[col_labels == lbl]
    ))
    label_to_col = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    return {r.region_id: label_to_col[col_labels[i]] for i, r in enumerate(regions)}


def _sort_reading_order(regions: List[LayoutRegion], page_width: int) -> List[LayoutRegion]:
    """
    Sort regions into natural reading order: top-to-bottom within each
    column, left column before right column.
    """
    col_map = _detect_columns(regions, page_width)
    for region in regions:
        region.column_index = col_map.get(region.region_id, 0)

    return sorted(
        regions,
        key=lambda r: (r.column_index, r.bounding_box.y_min, r.bounding_box.x_min),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class LayoutAnalyser:
    """
    Converts raw OCR output (list of TextBlocks) into structured layout
    regions using spatial clustering and trained classification.

    Classification priority:
        1. LightGBM model trained on DocBank (if model file exists)
        2. Heuristic rules (fallback, same as original implementation)
    """

    def __init__(self):
        from app.classifier.classifier import get_classifier
        self._classifier = get_classifier()
        if self._classifier.is_available:
            logger.info("LayoutAnalyser: using trained LightGBM classifier")
        else:
            logger.info("LayoutAnalyser: trained model not found — using heuristic fallback")

    def analyse(self, ocr_output: RawOCROutput, page_number: int) -> PageResult:
        """
        Analyse layout for a single page.

        Parameters
        ----------
        ocr_output : RawOCROutput
            Output from the OCR pipeline containing TextBlocks.
        page_number : int
            1-indexed page number.

        Returns
        -------
        PageResult
            Structured page with LayoutRegions in reading order.
        """
        blocks = ocr_output.text_blocks
        page_w = ocr_output.page_width
        page_h = ocr_output.page_height

        if not blocks:
            return PageResult(
                page_number=page_number,
                width_px=page_w,
                height_px=page_h,
            )

        try:
            regions = self._build_regions(blocks, page_w, page_h, page_number)
            ordered = _sort_reading_order(regions, page_w)
        except Exception as exc:
            raise LayoutAnalysisError(f"Layout analysis failed on page {page_number}: {exc}") from exc

        return PageResult(
            page_number=page_number,
            width_px=page_w,
            height_px=page_h,
            regions=ordered,
            raw_text_blocks=blocks,
        )

    def _build_regions(
        self,
        blocks: List[TextBlock],
        page_width: int,
        page_height: int,
        page_number: int,
    ) -> List[LayoutRegion]:
        """Cluster blocks and classify each cluster."""
        cluster_labels = _cluster_blocks(blocks)
        median_h = _median_char_height(blocks)

        # Group blocks by cluster label
        clusters: Dict[int, List[TextBlock]] = {}
        for block, lbl in zip(blocks, cluster_labels):
            clusters.setdefault(int(lbl), []).append(block)

        regions: List[LayoutRegion] = []
        for cluster_id, cluster_blocks in clusters.items():
            # Sort blocks within cluster by reading order (top-left → bottom-right)
            cluster_blocks = sorted(
                cluster_blocks,
                key=lambda b: (b.bounding_box.y_min, b.bounding_box.x_min),
            )

            # Use trained classifier if available, else heuristic fallback
            if self._classifier.is_available:
                label, confidence = self._classifier.predict(
                    cluster_blocks, page_width, page_height, median_h
                )
            else:
                label, confidence = _classify_region(
                    cluster_blocks, page_width, page_height, median_h
                )
            bbox = _enclosing_bbox(cluster_blocks)
            avg_h = float(np.mean([_estimate_char_height(b) for b in cluster_blocks]))
            full_text = " ".join(b.text for b in cluster_blocks)

            region = LayoutRegion(
                region_id=f"p{page_number}_r{cluster_id}_{uuid.uuid4().hex[:6]}",
                label=label,
                confidence=confidence,
                bounding_box=bbox,
                text_blocks=cluster_blocks,
                text=full_text,
                page_number=page_number,
                avg_char_height=round(avg_h, 2),
            )
            regions.append(region)

        return regions
