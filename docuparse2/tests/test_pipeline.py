"""
Test suite for DocuParse.

Tests are structured to run without PaddleOCR models downloaded
(heavy OCR tests are marked with @pytest.mark.integration).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.models.schemas import BoundingBox, RegionLabel, TextBlock
from app.services.layout_analyser import (
    LayoutAnalyser,
    _classify_region,
    _cluster_blocks,
    _has_column_alignment,
    _median_char_height,
)
from app.services.ocr_pipeline import _quad_to_axis_aligned_bbox


# ─────────────────────────────────────────────────────────────────────────────
# BoundingBox
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundingBox:
    def test_dimensions(self):
        bb = BoundingBox(x_min=10, y_min=20, x_max=110, y_max=70)
        assert bb.width == 100
        assert bb.height == 50
        assert bb.area == 5000

    def test_center(self):
        bb = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=200)
        assert bb.center == (50.0, 100.0)

    def test_invalid_raises(self):
        with pytest.raises(Exception):
            BoundingBox(x_min=100, y_min=0, x_max=50, y_max=50)  # x_max < x_min


# ─────────────────────────────────────────────────────────────────────────────
# OCR Pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestQuadToAxisAligned:
    def test_horizontal_quad(self):
        quad = np.array([[10, 20], [110, 20], [110, 50], [10, 50]], dtype=np.float32)
        bb = _quad_to_axis_aligned_bbox(quad)
        assert bb.x_min == pytest.approx(10.0)
        assert bb.y_min == pytest.approx(20.0)
        assert bb.x_max == pytest.approx(110.0)
        assert bb.y_max == pytest.approx(50.0)

    def test_rotated_quad(self):
        # Diamond shape — axis-aligned box should envelope it
        quad = np.array([[50, 0], [100, 50], [50, 100], [0, 50]], dtype=np.float32)
        bb = _quad_to_axis_aligned_bbox(quad)
        assert bb.x_min == pytest.approx(0.0)
        assert bb.x_max == pytest.approx(100.0)
        assert bb.y_min == pytest.approx(0.0)
        assert bb.y_max == pytest.approx(100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Layout Analyser — clustering
# ─────────────────────────────────────────────────────────────────────────────

def _make_block(x_min, y_min, x_max, y_max, text="sample", confidence=0.95) -> TextBlock:
    return TextBlock(
        text=text,
        confidence=confidence,
        bounding_box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
        page_number=1,
    )


class TestClustering:
    def test_single_block_forms_one_cluster(self):
        blocks = [_make_block(0, 0, 100, 20)]
        labels = _cluster_blocks(blocks)
        assert len(set(labels)) == 1

    def test_widely_separated_blocks_form_separate_clusters(self):
        blocks = [
            _make_block(0, 0, 100, 20),
            _make_block(0, 500, 100, 520),
        ]
        labels = _cluster_blocks(blocks)
        assert len(set(labels)) == 2

    def test_adjacent_blocks_cluster_together(self):
        blocks = [
            _make_block(0, 100, 200, 120),
            _make_block(210, 102, 400, 122),  # Same line, slightly right
        ]
        labels = _cluster_blocks(blocks)
        # Should cluster into same group (within eps_x)
        assert len(set(labels)) == 1

    def test_no_blocks_returns_empty(self):
        labels = _cluster_blocks([])
        assert len(labels) == 0


class TestRegionClassification:
    def _blocks_at_height(self, y, height, text="word", n=5):
        return [_make_block(i * 120, y, (i + 1) * 120 - 5, y + height, text) for i in range(n)]

    def test_large_text_classified_as_heading(self):
        # Median height = 12, heading height = 36 → ratio 3.0 > threshold 1.4
        body_blocks = self._blocks_at_height(200, 12, "body", n=20)
        heading_blocks = self._blocks_at_height(50, 36, "Title", n=3)
        all_blocks = body_blocks + heading_blocks
        median = _median_char_height(all_blocks)

        label, conf = _classify_region(heading_blocks, 800, 1100, median)
        assert label == RegionLabel.HEADING
        assert conf > 0.7

    def test_normal_text_classified_as_paragraph(self):
        blocks = self._blocks_at_height(200, 12, "normal text here", n=6)
        median = _median_char_height(blocks)
        label, _ = _classify_region(blocks, 800, 1100, median)
        assert label == RegionLabel.PARAGRAPH

    def test_top_margin_classified_as_header(self):
        # y_center at 5% of page height → HEADER
        blocks = [_make_block(0, 20, 400, 38, "Page 1 of 10")]
        median = 12.0
        label, _ = _classify_region(blocks, 800, 1100, median)
        assert label == RegionLabel.HEADER

    def test_bottom_margin_classified_as_footer(self):
        blocks = [_make_block(0, 1070, 400, 1090, "Confidential")]
        median = 12.0
        label, _ = _classify_region(blocks, 800, 1100, median)
        assert label == RegionLabel.FOOTER


class TestColumnAlignment:
    def test_two_column_table_detected(self):
        # Two x-columns, three y-rows
        blocks = [
            _make_block(10, 100, 150, 120, "Name"),
            _make_block(200, 100, 350, 120, "Age"),
            _make_block(10, 130, 150, 150, "Alice"),
            _make_block(200, 130, 350, 150, "30"),
            _make_block(10, 160, 150, 180, "Bob"),
            _make_block(200, 160, 350, 180, "25"),
        ]
        assert _has_column_alignment(blocks, tolerance=10.0) is True

    def test_single_column_not_detected_as_table(self):
        blocks = [_make_block(50, y, 300, y + 20, "text") for y in range(100, 500, 30)]
        assert _has_column_alignment(blocks, tolerance=10.0) is False


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests (require PaddleOCR models — skip in CI)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestOCRPipelineIntegration:
    """
    These tests require PaddleOCR model weights to be downloaded.
    Run with: pytest -m integration
    """

    def test_pipeline_runs_on_synthetic_image(self):
        from app.services.ocr_pipeline import OCRPipeline

        pipeline = OCRPipeline.get_instance()
        # White image — no text, should return empty blocks
        blank = np.ones((400, 600, 3), dtype=np.uint8) * 255
        result = pipeline.run(blank, page_number=1)
        assert isinstance(result.text_blocks, list)
        assert result.page_width == 600
        assert result.page_height == 400
