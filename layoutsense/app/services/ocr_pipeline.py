"""
Three-Stage OCR Pipeline
========================
Implements the PP-OCR architecture described in Du et al. (2020):
"PP-OCR: A Practical Ultra Lightweight OCR System"

Stage 1 — Text Detection
    Uses Differentiable Binarization (DB) to produce binary maps from
    which text bounding boxes are extracted. DB is preferred over older
    methods (e.g., EAST, CRAFT) because its post-processing is integrated
    into the network, enabling real-time inference on CPU.

Stage 2 — Direction Classification
    A lightweight MobileNetV3 classifier determines whether each detected
    text region is upright (0°) or rotated (180°). This is necessary
    for documents scanned upside-down or containing mixed orientations.

Stage 3 — Text Recognition
    SVTR_LCNet (Simple Visual Text Recognition + LCNet backbone) converts
    each rectified text region into a string. PaddleOCR's CTC-based decoder
    is used for sequence-to-sequence alignment.

PaddleOCR is used as the reference implementation backbone. The pipeline
is structured to mirror the paper's three-stage separation rather than
calling PaddleOCR as a black box, making each stage inspectable.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.core.exceptions import OCRPipelineError
from app.models.schemas import BoundingBox, TextBlock

logger = logging.getLogger(__name__)

# PaddleOCR import is deferred to first use to avoid slowing imports
# and to allow the service to start even if the model weights are being downloaded.
_paddleocr_lock = threading.Lock()
_paddle_instance: Optional[Any] = None


@dataclass
class DetectionResult:
    """Output of Stage 1: a list of quadrilateral bounding boxes."""
    boxes: List[np.ndarray]        # Each box: shape (4, 2) — four corner points
    scores: List[float]
    page_width: int
    page_height: int


@dataclass
class ClassificationResult:
    """Output of Stage 2: orientation label per detected box."""
    labels: List[str]              # "0" or "180"
    scores: List[float]


@dataclass
class RecognitionResult:
    """Output of Stage 3: recognized text and confidence per box."""
    texts: List[str]
    confidences: List[float]


@dataclass
class RawOCROutput:
    """Aggregated output of all three pipeline stages for one page."""
    text_blocks: List[TextBlock] = field(default_factory=list)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    page_width: int = 0
    page_height: int = 0


def _get_paddle_ocr():
    """
    Lazily initialise PaddleOCR as a singleton.
    Thread-safe via a module-level lock.
    """
    global _paddle_instance
    if _paddle_instance is not None:
        return _paddle_instance

    with _paddleocr_lock:
        if _paddle_instance is not None:  # Double-checked locking
            return _paddle_instance

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise OCRPipelineError(
                "PaddleOCR is not installed. Run: pip install paddleocr"
            ) from exc

        logger.info(
            "Initialising PaddleOCR (lang=%s, GPU=%s, det_algo=%s, rec_algo=%s)",
            settings.OCR_LANG,
            settings.OCR_USE_GPU,
            settings.OCR_DET_ALGORITHM,
            settings.OCR_REC_ALGORITHM,
        )

        _paddle_instance = PaddleOCR(
            use_angle_cls=settings.OCR_USE_ANGLE_CLS,
            lang=settings.OCR_LANG,
            use_gpu=settings.OCR_USE_GPU,
            det_algorithm=settings.OCR_DET_ALGORITHM,
            det_db_thresh=settings.OCR_DET_DB_THRESH,
            det_db_box_thresh=settings.OCR_DET_DB_BOX_THRESH,
            det_db_unclip_ratio=settings.OCR_DET_DB_UNCLIP_RATIO,
            cls_thresh=settings.OCR_CLS_THRESH,
            rec_algorithm=settings.OCR_REC_ALGORITHM,
            show_log=False,
        )
        logger.info("PaddleOCR initialised.")
    return _paddle_instance


def _quad_to_axis_aligned_bbox(quad: np.ndarray) -> BoundingBox:
    """
    Convert a quadrilateral (4 corner points from DB detector) to an
    axis-aligned bounding box.

    The DB detector returns rotated quads; for layout analysis we work
    with axis-aligned boxes because DBSCAN clustering operates on
    projections onto the X and Y axes.
    """
    xs = quad[:, 0]
    ys = quad[:, 1]
    return BoundingBox(
        x_min=float(np.min(xs)),
        y_min=float(np.min(ys)),
        x_max=float(np.max(xs)),
        y_max=float(np.max(ys)),
    )


def _estimate_angle_from_quad(quad: np.ndarray) -> float:
    """
    Estimate text line angle from the bottom edge of the detected quad.
    Returns angle in degrees; 0° = horizontal.
    """
    # Bottom edge: points[2] → points[3] in PaddleOCR's quad ordering
    dx = quad[2][0] - quad[3][0]
    dy = quad[2][1] - quad[3][1]
    angle_rad = np.arctan2(dy, dx)
    return float(np.degrees(angle_rad))


class OCRPipeline:
    """
    Orchestrates the three PP-OCR stages in sequence.

    Design decision: rather than calling paddle_ocr.ocr() as a single
    black-box call, we expose each stage individually so that:
    (a) timing data per stage is captured,
    (b) intermediate outputs can be inspected for debugging, and
    (c) individual stages can be swapped for alternative implementations.
    """

    _instance: Optional["OCRPipeline"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._ocr = None  # Deferred until first run

    @classmethod
    def get_instance(cls) -> "OCRPipeline":
        """Return the singleton OCRPipeline, initialising if needed."""
        if cls._instance is not None:
            return cls._instance
        with cls._instance_lock:
            if cls._instance is None:
                instance = cls()
                instance._ocr = _get_paddle_ocr()
                cls._instance = instance
        return cls._instance

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1: Text Detection
    # ─────────────────────────────────────────────────────────────────────────

    def _stage1_detect(self, image: np.ndarray) -> DetectionResult:
        """
        Run the DB text detector on the input image.

        DB (Differentiable Binarization) predicts a probability map for
        each pixel's likelihood of belonging to a text region. A fast
        post-processing step thresholds and expands the binary map
        using the configured unclip ratio to produce final bounding boxes.

        Returns quadrilateral boxes because DB naturally detects rotated text.
        """
        h, w = image.shape[:2]
        t0 = time.perf_counter()

        try:
            # We call the internal detection method directly to keep stages separate
            detection_result = self._ocr.text_detector(image)
        except Exception as exc:
            raise OCRPipelineError(f"Stage 1 (detection) failed: {exc}") from exc

        elapsed = time.perf_counter() - t0
        logger.debug("Stage 1 complete in %.3fs — %d boxes detected", elapsed, len(detection_result[0]) if detection_result[0] is not None else 0)

        boxes, scores = [], []
        if detection_result[0] is not None:
            for item in detection_result[0]:
                boxes.append(item.astype(np.float32))
                # DB scores are embedded in the result array; use placeholder if unavailable
                scores.append(1.0)

        return DetectionResult(boxes=boxes, scores=scores, page_width=w, page_height=h)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2: Direction Classification
    # ─────────────────────────────────────────────────────────────────────────

    def _stage2_classify(
        self, image: np.ndarray, detection: DetectionResult
    ) -> ClassificationResult:
        """
        Classify the text orientation for each detected bounding box.

        Uses a lightweight MobileNetV3 classifier trained on 0°/180° labels.
        For documents without rotation, this stage adds minimal overhead
        (~2ms per region on CPU) but prevents recognition errors on rotated scans.
        """
        if not settings.OCR_USE_ANGLE_CLS or not detection.boxes:
            return ClassificationResult(
                labels=["0"] * len(detection.boxes),
                scores=[1.0] * len(detection.boxes),
            )

        t0 = time.perf_counter()
        try:
            # Crop each detected region from the image for classification
            crops = self._crop_regions(image, detection.boxes)
            if not crops:
                return ClassificationResult(labels=[], scores=[])

            cls_result, _ = self._ocr.text_classifier(crops)
        except Exception as exc:
            logger.warning("Stage 2 (classification) failed, defaulting to 0°: %s", exc)
            return ClassificationResult(
                labels=["0"] * len(detection.boxes),
                scores=[1.0] * len(detection.boxes),
            )

        elapsed = time.perf_counter() - t0
        logger.debug("Stage 2 complete in %.3fs", elapsed)

        labels = [r[0] for r in cls_result] if cls_result else ["0"] * len(detection.boxes)
        scores = [float(r[1]) for r in cls_result] if cls_result else [1.0] * len(detection.boxes)
        return ClassificationResult(labels=labels, scores=scores)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3: Text Recognition
    # ─────────────────────────────────────────────────────────────────────────

    def _stage3_recognise(
        self,
        image: np.ndarray,
        detection: DetectionResult,
        classification: ClassificationResult,
    ) -> RecognitionResult:
        """
        Recognise text within each detected and oriented bounding box.

        SVTR_LCNet encodes the cropped region into a sequence of features
        via a hybrid CNN-Transformer backbone, then decodes with CTC.
        The 180°-labelled crops are flipped before recognition.
        """
        if not detection.boxes:
            return RecognitionResult(texts=[], confidences=[])

        t0 = time.perf_counter()
        try:
            crops = self._crop_and_orient_regions(image, detection.boxes, classification.labels)
            if not crops:
                return RecognitionResult(texts=[], confidences=[])

            rec_result, _ = self._ocr.text_recognizer(crops)
        except Exception as exc:
            raise OCRPipelineError(f"Stage 3 (recognition) failed: {exc}") from exc

        elapsed = time.perf_counter() - t0
        logger.debug("Stage 3 complete in %.3fs — %d texts recognised", elapsed, len(rec_result))

        texts = [r[0] for r in rec_result] if rec_result else []
        confidences = [float(r[1]) for r in rec_result] if rec_result else []
        return RecognitionResult(texts=texts, confidences=confidences)

    # ─────────────────────────────────────────────────────────────────────────
    # Public pipeline entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, image: np.ndarray, page_number: int = 1) -> RawOCROutput:
        """
        Execute the full 3-stage pipeline on a single page image.

        Parameters
        ----------
        image : np.ndarray
            RGB image array (H, W, 3).
        page_number : int
            1-indexed page number (used to populate TextBlock metadata).

        Returns
        -------
        RawOCROutput
            All detected TextBlocks plus per-stage timing data.
        """
        if self._ocr is None:
            self._ocr = _get_paddle_ocr()

        total_start = time.perf_counter()
        stage_timings: Dict[str, float] = {}

        # ── Stage 1: Detect ──────────────────────────────────────────────────
        t = time.perf_counter()
        detection = self._stage1_detect(image)
        stage_timings["stage1_detection_s"] = round(time.perf_counter() - t, 4)

        if not detection.boxes:
            logger.info("Page %d: no text regions detected.", page_number)
            return RawOCROutput(
                stage_timings=stage_timings,
                page_width=detection.page_width,
                page_height=detection.page_height,
            )

        # ── Stage 2: Classify ─────────────────────────────────────────────────
        t = time.perf_counter()
        classification = self._stage2_classify(image, detection)
        stage_timings["stage2_classification_s"] = round(time.perf_counter() - t, 4)

        # ── Stage 3: Recognise ─────────────────────────────────────────────────
        t = time.perf_counter()
        recognition = self._stage3_recognise(image, detection, classification)
        stage_timings["stage3_recognition_s"] = round(time.perf_counter() - t, 4)

        stage_timings["total_s"] = round(time.perf_counter() - total_start, 4)

        # ── Assemble TextBlocks ───────────────────────────────────────────────
        text_blocks: List[TextBlock] = []
        n = min(len(detection.boxes), len(recognition.texts))

        for i in range(n):
            text = recognition.texts[i].strip() if i < len(recognition.texts) else ""
            conf = recognition.confidences[i] if i < len(recognition.confidences) else 0.0
            angle_label = classification.labels[i] if i < len(classification.labels) else "0"

            if not text:
                continue

            bbox = _quad_to_axis_aligned_bbox(detection.boxes[i])
            quad_angle = _estimate_angle_from_quad(detection.boxes[i])
            # The classifier may have flipped the crop; account for that
            orientation_degrees = 180.0 if angle_label == "180" else 0.0

            text_blocks.append(
                TextBlock(
                    text=text,
                    confidence=conf,
                    bounding_box=bbox,
                    angle=round(quad_angle + orientation_degrees, 2),
                    page_number=page_number,
                )
            )

        logger.info(
            "Page %d: %d text blocks extracted in %.3fs",
            page_number,
            len(text_blocks),
            stage_timings["total_s"],
        )

        return RawOCROutput(
            text_blocks=text_blocks,
            stage_timings=stage_timings,
            page_width=detection.page_width,
            page_height=detection.page_height,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _crop_regions(image: np.ndarray, boxes: List[np.ndarray]) -> List[np.ndarray]:
        """Crop axis-aligned regions from the image for each detected box."""
        crops = []
        h, w = image.shape[:2]
        for box in boxes:
            x_min = max(0, int(np.min(box[:, 0])))
            y_min = max(0, int(np.min(box[:, 1])))
            x_max = min(w, int(np.max(box[:, 0])))
            y_max = min(h, int(np.max(box[:, 1])))
            if x_max > x_min and y_max > y_min:
                crops.append(image[y_min:y_max, x_min:x_max])
        return crops

    @staticmethod
    def _crop_and_orient_regions(
        image: np.ndarray,
        boxes: List[np.ndarray],
        labels: List[str],
    ) -> List[np.ndarray]:
        """Crop regions and apply 180° flip for classifier-labelled rotated text."""
        crops = []
        h, w = image.shape[:2]
        for idx, box in enumerate(boxes):
            x_min = max(0, int(np.min(box[:, 0])))
            y_min = max(0, int(np.min(box[:, 1])))
            x_max = min(w, int(np.max(box[:, 0])))
            y_max = min(h, int(np.max(box[:, 1])))
            if x_max <= x_min or y_max <= y_min:
                continue
            crop = image[y_min:y_max, x_min:x_max]
            if idx < len(labels) and labels[idx] == "180":
                crop = np.rot90(crop, k=2)  # 180° rotation
            crops.append(crop)
        return crops
