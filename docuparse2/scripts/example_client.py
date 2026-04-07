#!/usr/bin/env python3
"""
DocuParse Client Example
========================
Demonstrates the full upload → parse → structure workflow.

Usage:
    python scripts/example_client.py path/to/document.pdf
    python scripts/example_client.py path/to/image.png --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def upload(base_url: str, filepath: Path) -> str:
    """Upload a document and return its document_id."""
    print(f"[1/3] Uploading '{filepath.name}'...")
    with open(filepath, "rb") as f:
        resp = requests.post(
            f"{base_url}/api/v1/upload",
            files={"file": (filepath.name, f, _guess_mime(filepath))},
            timeout=60,
        )
    resp.raise_for_status()
    data = resp.json()
    doc_id = data["document_id"]
    print(f"      ✓ document_id={doc_id}, pages={data['total_pages']}")
    return doc_id


def parse(base_url: str, doc_id: str) -> dict:
    """Run the OCR + layout pipeline and return the ParseResponse."""
    print(f"[2/3] Parsing (this may take a moment on first run)...")
    t0 = time.time()
    resp = requests.post(
        f"{base_url}/api/v1/parse",
        data={"document_id": doc_id},
        timeout=300,
    )
    resp.raise_for_status()
    elapsed = time.time() - t0
    data = resp.json()
    print(f"      ✓ {data['total_pages']} page(s) parsed in {data['processing_time_seconds']:.2f}s")
    print(f"      Pipeline: {data['pipeline_stages']['stage1_text_detection']}")

    total_regions = sum(len(p["regions"]) for p in data["pages"])
    print(f"      Detected {total_regions} layout region(s)")
    return data


def get_structure(base_url: str, doc_id: str) -> dict:
    """Fetch the condensed structure view."""
    print(f"[3/3] Fetching document structure...")
    resp = requests.get(f"{base_url}/api/v1/structure/{doc_id}", timeout=30)
    resp.raise_for_status()
    data = resp.json()

    print(f"\n{'─' * 60}")
    print(f"  Document: {data['filename']}")
    print(f"{'─' * 60}")
    print(f"  Headings  ({len(data['headings'])}):")
    for h in data["headings"]:
        print(f"    • {h}")
    print(f"\n  Paragraphs ({len(data['paragraphs'])}):")
    for p in data["paragraphs"][:3]:  # Show first 3
        print(f"    ▸ {p[:80]}{'...' if len(p) > 80 else ''}")
    if len(data["paragraphs"]) > 3:
        print(f"    ... ({len(data['paragraphs']) - 3} more)")
    print(f"\n  Tables    ({len(data['tables'])})")
    print(f"  Figures   ({len(data['figures'])})")
    print(f"{'─' * 60}\n")
    return data


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")


def main():
    parser = argparse.ArgumentParser(description="DocuParse example client")
    parser.add_argument("filepath", type=Path, help="Path to PDF or image file")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", type=Path, help="Save ParseResponse JSON to file")
    args = parser.parse_args()

    if not args.filepath.exists():
        print(f"Error: file not found: {args.filepath}", file=sys.stderr)
        sys.exit(1)

    try:
        doc_id = upload(args.base_url, args.filepath)
        parse_result = parse(args.base_url, doc_id)
        structure = get_structure(args.base_url, doc_id)

        if args.output:
            args.output.write_text(json.dumps(parse_result, indent=2))
            print(f"Full ParseResponse saved to {args.output}")

    except requests.exceptions.ConnectionError:
        print(f"\nError: could not connect to {args.base_url}", file=sys.stderr)
        print("Is the DocuParse API running? Try: docker compose up", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError as exc:
        print(f"\nHTTP error: {exc.response.status_code} — {exc.response.text}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
