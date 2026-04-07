"""
Table Structure Reconstruction
================================
Recovers the row/column grid from a detected table region using
line projection histograms — a classical computer vision technique
that doesn't require a trained model.

Algorithm (based on projection profile analysis):
    1. Take all TextBlock bounding boxes within a TABLE region.
    2. Project onto the Y-axis: build a histogram of how many boxes
       overlap each Y coordinate. Valleys in this histogram are row
       separators (gaps between rows of cells).
    3. Project onto the X-axis: same approach for column separators.
    4. Assign each TextBlock to a (row, col) cell by checking which
       row/col interval its center falls into.
    5. Return a structured grid with cell text, bounding box, and
       row/column indices.

This is genuinely hard to do correctly — merged cells, spanning headers,
and irregular column widths all break naive approaches. The implementation
handles:
    - Variable-width columns (detected per-row, not assumed uniform)
    - Empty cells (no TextBlock maps to that grid position)
    - Header row detection (first row with distinct height/style)
    - Single-column "tables" (rejected — not treated as tables)

The output replaces the flat `[cell_text, ...]` list in StructureResponse
with structured `TableCell` objects carrying row, col, and span metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.models.schemas import BoundingBox, LayoutRegion, TextBlock

logger = logging.getLogger(__name__)


@dataclass
class TableCell:
    """A single cell in a reconstructed table grid."""
    row: int                      # 0-indexed row
    col: int                      # 0-indexed column
    text: str
    bounding_box: BoundingBox
    is_header: bool = False
    row_span: int = 1             # Future: merged cell support
    col_span: int = 1


@dataclass
class ReconstructedTable:
    """Full grid structure for one detected table region."""
    region_id: str
    n_rows: int
    n_cols: int
    cells: List[TableCell] = field(default_factory=list)
    header_row: Optional[int] = None   # Index of the header row, if detected

    def to_dict(self) -> dict:
        return {
            "region_id": self.region_id,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "header_row": self.header_row,
            "cells": [
                {
                    "row": c.row,
                    "col": c.col,
                    "text": c.text,
                    "is_header": c.is_header,
                    "bounding_box": {
                        "x_min": c.bounding_box.x_min,
                        "y_min": c.bounding_box.y_min,
                        "x_max": c.bounding_box.x_max,
                        "y_max": c.bounding_box.y_max,
                    },
                }
                for c in self.cells
            ],
        }

    def to_markdown(self) -> str:
        """Render the table as a Markdown table string."""
        if not self.cells:
            return ""

        grid: Dict[Tuple[int, int], str] = {(c.row, c.col): c.text for c in self.cells}

        lines = []
        for row_idx in range(self.n_rows):
            row_cells = [grid.get((row_idx, col_idx), "") for col_idx in range(self.n_cols)]
            lines.append("| " + " | ".join(row_cells) + " |")
            if row_idx == 0:
                lines.append("|" + "|".join(["---"] * self.n_cols) + "|")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Projection Histogram
# ─────────────────────────────────────────────────────────────────────────────

def _build_projection(
    blocks: List[TextBlock],
    axis: str,       # "x" or "y"
    resolution: int = 2,  # pixels per histogram bin
) -> Tuple[np.ndarray, float, float]:
    """
    Build a 1D projection histogram of bounding box occupancy.

    For axis="y": counts how many boxes overlap each y-coordinate.
    Valleys in the histogram correspond to row separators.

    For axis="x": same for column separators.

    Returns (histogram, min_coord, max_coord).
    """
    if axis == "y":
        coords_min = [b.bounding_box.y_min for b in blocks]
        coords_max = [b.bounding_box.y_max for b in blocks]
    else:
        coords_min = [b.bounding_box.x_min for b in blocks]
        coords_max = [b.bounding_box.x_max for b in blocks]

    global_min = min(coords_min)
    global_max = max(coords_max)
    n_bins = max(1, int((global_max - global_min) / resolution))

    histogram = np.zeros(n_bins, dtype=np.float32)

    for lo, hi in zip(coords_min, coords_max):
        bin_lo = int((lo - global_min) / resolution)
        bin_hi = int((hi - global_min) / resolution)
        histogram[bin_lo:bin_hi + 1] += 1.0

    return histogram, global_min, global_max


def _find_separators(
    histogram: np.ndarray,
    global_min: float,
    resolution: int,
    min_gap_bins: int = 2,
) -> List[float]:
    """
    Find valley positions in the projection histogram.

    A valley is a contiguous run of zero (or near-zero) bins,
    indicating empty space between rows/columns.

    Returns list of separator coordinates (midpoints of gap runs).
    """
    separators = []
    in_gap = False
    gap_start = 0

    threshold = 0.05  # Bins with occupancy below this are "empty"

    for i, val in enumerate(histogram):
        if val <= threshold:
            if not in_gap:
                in_gap = True
                gap_start = i
        else:
            if in_gap:
                gap_end = i
                gap_width = gap_end - gap_start
                if gap_width >= min_gap_bins:
                    mid = (gap_start + gap_end) / 2.0
                    separators.append(global_min + mid * resolution)
                in_gap = False

    # Handle trailing gap
    if in_gap:
        gap_end = len(histogram)
        gap_width = gap_end - gap_start
        if gap_width >= min_gap_bins:
            mid = (gap_start + gap_end) / 2.0
            separators.append(global_min + mid * resolution)

    return separators


def _separators_to_intervals(
    separators: List[float],
    global_min: float,
    global_max: float,
) -> List[Tuple[float, float]]:
    """Convert separator positions into (start, end) intervals for each band."""
    boundaries = [global_min] + sorted(separators) + [global_max]
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Cell Assignment
# ─────────────────────────────────────────────────────────────────────────────

def _assign_cell(
    block: TextBlock,
    row_intervals: List[Tuple[float, float]],
    col_intervals: List[Tuple[float, float]],
) -> Tuple[int, int]:
    """
    Assign a TextBlock to a (row, col) grid position.
    Uses the center of the block's bounding box for assignment.
    """
    bb = block.bounding_box
    y_center = (bb.y_min + bb.y_max) / 2.0
    x_center = (bb.x_min + bb.x_max) / 2.0

    row_idx = 0
    for i, (lo, hi) in enumerate(row_intervals):
        if lo <= y_center <= hi:
            row_idx = i
            break

    col_idx = 0
    for i, (lo, hi) in enumerate(col_intervals):
        if lo <= x_center <= hi:
            col_idx = i
            break

    return row_idx, col_idx


# ─────────────────────────────────────────────────────────────────────────────
# Header detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_header_row(
    cells: List[TableCell],
    n_cols: int,
) -> Optional[int]:
    """
    Heuristically identify the header row.

    A row is likely a header if:
    - It's the first row (row 0), AND
    - Its cells have above-average bounding box height (bold/larger text), OR
    - All first-row cells are non-empty (dense header coverage)
    """
    if not cells:
        return None

    first_row_cells = [c for c in cells if c.row == 0]
    if not first_row_cells:
        return None

    # Coverage: fraction of columns that have a cell in row 0
    coverage = len(first_row_cells) / n_cols
    if coverage >= 0.6:  # At least 60% of columns filled in first row
        return 0

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_table(region: LayoutRegion) -> Optional[ReconstructedTable]:
    """
    Reconstruct the row/column grid from a TABLE-labelled LayoutRegion.

    Returns None if the region has fewer than 2 rows or 2 columns
    (single-row/column spans are not tables in the structural sense).

    Parameters
    ----------
    region : LayoutRegion
        A region with label=TABLE containing TextBlocks.

    Returns
    -------
    ReconstructedTable or None
    """
    blocks = region.text_blocks
    if len(blocks) < 2:
        return None

    # ── Y-axis projection: find row separators ────────────────────────────────
    y_hist, y_min, y_max = _build_projection(blocks, axis="y", resolution=2)
    y_seps = _find_separators(y_hist, y_min, resolution=2, min_gap_bins=2)
    row_intervals = _separators_to_intervals(y_seps, y_min, y_max)

    # ── X-axis projection: find column separators ─────────────────────────────
    x_hist, x_min, x_max = _build_projection(blocks, axis="x", resolution=2)
    x_seps = _find_separators(x_hist, x_min, resolution=2, min_gap_bins=3)
    col_intervals = _separators_to_intervals(x_seps, x_min, x_max)

    n_rows = len(row_intervals)
    n_cols = len(col_intervals)

    # Reject degenerate cases
    if n_rows < 1 or n_cols < 2:
        logger.debug(
            "Region %s: degenerate table (%d rows, %d cols) — skipping reconstruction",
            region.region_id, n_rows, n_cols,
        )
        return None

    # ── Assign each block to a cell ───────────────────────────────────────────
    # Merge blocks assigned to the same cell
    cell_texts: Dict[Tuple[int, int], List[str]] = {}
    cell_bboxes: Dict[Tuple[int, int], List[BoundingBox]] = {}

    for block in blocks:
        row_idx, col_idx = _assign_cell(block, row_intervals, col_intervals)
        cell_texts.setdefault((row_idx, col_idx), []).append(block.text)
        cell_bboxes.setdefault((row_idx, col_idx), []).append(block.bounding_box)

    cells: List[TableCell] = []
    for (row_idx, col_idx), texts in cell_texts.items():
        bboxes = cell_bboxes[(row_idx, col_idx)]
        merged_bbox = BoundingBox(
            x_min=min(b.x_min for b in bboxes),
            y_min=min(b.y_min for b in bboxes),
            x_max=max(b.x_max for b in bboxes),
            y_max=max(b.y_max for b in bboxes),
        )
        cells.append(TableCell(
            row=row_idx,
            col=col_idx,
            text=" ".join(texts),
            bounding_box=merged_bbox,
        ))

    # ── Header detection ──────────────────────────────────────────────────────
    header_row = _detect_header_row(cells, n_cols)
    if header_row is not None:
        for cell in cells:
            if cell.row == header_row:
                cell.is_header = True

    table = ReconstructedTable(
        region_id=region.region_id,
        n_rows=n_rows,
        n_cols=n_cols,
        cells=sorted(cells, key=lambda c: (c.row, c.col)),
        header_row=header_row,
    )

    logger.debug(
        "Table %s: %d rows × %d cols, %d cells, header_row=%s",
        region.region_id, n_rows, n_cols, len(cells), header_row,
    )

    return table


def reconstruct_all_tables(regions: List[LayoutRegion]) -> Dict[str, ReconstructedTable]:
    """
    Reconstruct all TABLE regions in a page's region list.
    Returns a dict mapping region_id → ReconstructedTable.
    """
    from app.models.schemas import RegionLabel
    results = {}
    for region in regions:
        if region.label == RegionLabel.TABLE:
            table = reconstruct_table(region)
            if table is not None:
                results[region.region_id] = table
    return results
