"""
Pydantic v2 models for API request/response contracts.

These schemas mirror the conceptual output of a LayoutLM-style pipeline:
each detected region carries both its geometric footprint (bounding box)
and its semantic label (heading, paragraph, table, figure, caption).
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class RegionLabel(str, Enum):
    """Semantic labels assigned during layout analysis."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    HEADER = "header"       # Page header (top margin content)
    FOOTER = "footer"       # Page footer (bottom margin content)
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    """
    Axis-aligned bounding box in pixel coordinates.
    Origin (0, 0) is the top-left corner of the page image.
    """
    x_min: float = Field(..., ge=0, description="Left edge (pixels)")
    y_min: float = Field(..., ge=0, description="Top edge (pixels)")
    x_max: float = Field(..., ge=0, description="Right edge (pixels)")
    y_max: float = Field(..., ge=0, description="Bottom edge (pixels)")

    @field_validator("x_max")
    @classmethod
    def x_max_gt_x_min(cls, v: float, info) -> float:
        if "x_min" in info.data and v <= info.data["x_min"]:
            raise ValueError("x_max must be greater than x_min")
        return v

    @field_validator("y_max")
    @classmethod
    def y_max_gt_y_min(cls, v: float, info) -> float:
        if "y_min" in info.data and v <= info.data["y_min"]:
            raise ValueError("y_max must be greater than y_min")
        return v

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)


class TextBlock(BaseModel):
    """
    A single OCR result: the smallest unit of detected text.
    Corresponds to one output line from PaddleOCR's recognition stage.
    """
    text: str = Field(..., description="Recognized text string")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Recognition confidence score")
    bounding_box: BoundingBox
    angle: Optional[float] = Field(
        default=0.0,
        description="Text orientation in degrees (from direction classifier, Stage 2)"
    )
    page_number: int = Field(default=1, ge=1)


class LayoutRegion(BaseModel):
    """
    A semantically labelled region of the document, produced by
    the LayoutLM-inspired spatial clustering and classification stage.
    Groups one or more TextBlocks that belong to the same logical element.
    """
    region_id: str = Field(..., description="Unique region identifier (e.g., 'page1_region_3')")
    label: RegionLabel
    confidence: float = Field(..., ge=0.0, le=1.0, description="Label assignment confidence")
    bounding_box: BoundingBox
    text_blocks: List[TextBlock] = Field(default_factory=list)
    text: str = Field(..., description="Concatenated text of all blocks in this region")
    page_number: int = Field(default=1, ge=1)

    # Spatial feature metadata (used for debugging / downstream ML)
    avg_char_height: Optional[float] = Field(
        default=None,
        description="Average character height (proxy for font size) in pixels"
    )
    column_index: Optional[int] = Field(
        default=None,
        description="Detected column index for multi-column layouts"
    )


class PageResult(BaseModel):
    """All layout regions detected on a single page."""
    page_number: int
    width_px: int
    height_px: int
    regions: List[LayoutRegion] = Field(default_factory=list)
    raw_text_blocks: List[TextBlock] = Field(
        default_factory=list,
        description="Pre-clustering OCR output (Stage 3 output, pre-layout)"
    )


class ParseResponse(BaseModel):
    """Top-level response from /parse — full pipeline output for a document."""
    document_id: str
    filename: str
    total_pages: int
    pages: List[PageResult]
    processing_time_seconds: float
    pipeline_stages: dict = Field(
        default_factory=lambda: {
            "stage1_text_detection": "DB (Differentiable Binarization)",
            "stage2_direction_classification": "MobileNetV3",
            "stage3_text_recognition": "SVTR_LCNet",
            "layout_analysis": "DBSCAN spatial clustering + heuristic classification",
        }
    )


class StructureResponse(BaseModel):
    """Condensed structure extraction — headings, paragraphs, tables only."""
    document_id: str
    filename: str
    headings: List[str] = Field(default_factory=list)
    paragraphs: List[str] = Field(default_factory=list)
    tables: List[List[str]] = Field(
        default_factory=list,
        description="Each entry is a list of cell texts for one detected table region"
    )
    figures: List[BoundingBox] = Field(
        default_factory=list,
        description="Bounding boxes of detected figure regions (no text)"
    )
    reading_order: List[str] = Field(
        default_factory=list,
        description="All region IDs in top-to-bottom, left-to-right reading order"
    )


class UploadResponse(BaseModel):
    """Response from /upload — acknowledges receipt and stores document."""
    document_id: str
    filename: str
    size_bytes: int
    mime_type: str
    total_pages: int
    message: str = "Document uploaded successfully. Use /parse to run the OCR pipeline."


class ErrorResponse(BaseModel):
    """Standard error envelope."""
    error: str
    detail: Optional[str] = None
    doc_url: str = "https://docuparse.example.com/docs"


# ─────────────────────────────────────────────────────────────────────────────
# Table reconstruction schemas (Part 5)
# ─────────────────────────────────────────────────────────────────────────────

class TableCellSchema(BaseModel):
    """Structured cell in a reconstructed table grid."""
    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)
    text: str
    is_header: bool = False
    row_span: int = 1
    col_span: int = 1
    bounding_box: BoundingBox


class ReconstructedTableSchema(BaseModel):
    """Full grid structure for one TABLE region."""
    region_id: str
    n_rows: int
    n_cols: int
    header_row: Optional[int] = None
    cells: List[TableCellSchema] = Field(default_factory=list)
    markdown: Optional[str] = Field(
        default=None,
        description="Markdown table representation"
    )


class StructureResponseV2(BaseModel):
    """
    Enriched structure response — tables are structured grids, not flat lists.
    Returned by GET /structure/{document_id}?v=2
    """
    document_id: str
    filename: str
    headings: List[str] = Field(default_factory=list)
    paragraphs: List[str] = Field(default_factory=list)
    tables: List[ReconstructedTableSchema] = Field(default_factory=list)
    figures: List[BoundingBox] = Field(default_factory=list)
    reading_order: List[str] = Field(default_factory=list)
