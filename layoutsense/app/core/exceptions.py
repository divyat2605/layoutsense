"""
Domain-specific exceptions for DocuParse.
Using typed exceptions rather than bare RuntimeError allows callers
to handle specific failure modes gracefully.
"""


class DocuParseError(Exception):
    """Base exception for all DocuParse errors."""


class UnsupportedFileTypeError(DocuParseError):
    """Raised when an uploaded file's MIME type is not supported."""

    def __init__(self, mime_type: str):
        super().__init__(f"Unsupported file type: '{mime_type}'. See /docs for accepted types.")
        self.mime_type = mime_type


class FileTooLargeError(DocuParseError):
    """Raised when an uploaded file exceeds the configured size limit."""

    def __init__(self, size_mb: float, limit_mb: int):
        super().__init__(f"File size {size_mb:.1f} MB exceeds the {limit_mb} MB limit.")
        self.size_mb = size_mb
        self.limit_mb = limit_mb


class PDFConversionError(DocuParseError):
    """Raised when PDF-to-image conversion fails."""


class OCRPipelineError(DocuParseError):
    """Raised when the OCR pipeline encounters an unrecoverable error."""


class LayoutAnalysisError(DocuParseError):
    """Raised when spatial layout analysis fails."""


class DocumentNotFoundError(DocuParseError):
    """Raised when a document ID does not exist in the cache/store."""

    def __init__(self, doc_id: str):
        super().__init__(f"Document '{doc_id}' not found. Upload it first via /upload.")
        self.doc_id = doc_id
