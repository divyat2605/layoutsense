"""
API endpoint tests using FastAPI's async test client.
These tests mock the DocumentProcessor to avoid requiring
PaddleOCR model weights in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.schemas import (
    BoundingBox,
    LayoutRegion,
    PageResult,
    ParseResponse,
    RegionLabel,
    TextBlock,
    UploadResponse,
)


def _make_mock_upload_response() -> UploadResponse:
    return UploadResponse(
        document_id="testdoc123abc456",
        filename="test.pdf",
        size_bytes=12345,
        mime_type="application/pdf",
        total_pages=1,
    )


def _make_mock_parse_response() -> ParseResponse:
    tb = TextBlock(
        text="Sample Heading",
        confidence=0.98,
        bounding_box=BoundingBox(x_min=50, y_min=80, x_max=400, y_max=130),
        page_number=1,
    )
    region = LayoutRegion(
        region_id="p1_r0_abc123",
        label=RegionLabel.HEADING,
        confidence=0.91,
        bounding_box=BoundingBox(x_min=50, y_min=80, x_max=400, y_max=130),
        text_blocks=[tb],
        text="Sample Heading",
        page_number=1,
        avg_char_height=42.0,
    )
    page = PageResult(
        page_number=1,
        width_px=800,
        height_px=1100,
        regions=[region],
        raw_text_blocks=[tb],
    )
    return ParseResponse(
        document_id="testdoc123abc456",
        filename="test.pdf",
        total_pages=1,
        pages=[page],
        processing_time_seconds=1.23,
    )


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


class TestHealthEndpoints:
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    async def test_ready_returns_200_or_503(self, client):
        resp = await client.get("/ready")
        assert resp.status_code in (200, 503)
        assert "checks" in resp.json()


class TestUploadEndpoint:
    async def test_upload_valid_pdf(self, client):
        mock_upload = _make_mock_upload_response()
        with patch(
            "app.api.routes.DocumentProcessor.upload", return_value=mock_upload
        ):
            resp = await client.post(
                "/api/v1/upload",
                files={"file": ("test.pdf", b"%PDF-1.4 fake content", "application/pdf")},
            )
        assert resp.status_code == 201
        body = resp.json()
        assert body["document_id"] == "testdoc123abc456"
        assert body["total_pages"] == 1

    async def test_upload_missing_file_returns_422(self, client):
        resp = await client.post("/api/v1/upload")
        assert resp.status_code == 422


class TestParseEndpoint:
    async def test_parse_returns_structured_response(self, client):
        mock_parse = _make_mock_parse_response()
        with patch(
            "app.api.routes.DocumentProcessor.parse", return_value=mock_parse
        ):
            resp = await client.post(
                "/api/v1/parse",
                data={"document_id": "testdoc123abc456"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_pages"] == 1
        assert len(body["pages"][0]["regions"]) == 1
        assert body["pages"][0]["regions"][0]["label"] == "heading"

    async def test_parse_missing_document_id_returns_422(self, client):
        resp = await client.post("/api/v1/parse")
        assert resp.status_code == 422


class TestStructureEndpoint:
    async def test_structure_requires_parsed_document(self, client):
        # Document not parsed yet → 409
        mock_record = {
            "filename": "test.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 1000,
            "total_pages": 1,
            "pages": [],
            "parsed": None,
        }
        with patch(
            "app.api.routes.DocumentProcessor.get_record", return_value=mock_record
        ):
            resp = await client.get("/api/v1/structure/testdoc123abc456")
        assert resp.status_code == 409

    async def test_structure_not_found(self, client):
        with patch(
            "app.api.routes.DocumentProcessor.get_record", return_value=None
        ):
            resp = await client.get("/api/v1/structure/nonexistent")
        assert resp.status_code == 404
