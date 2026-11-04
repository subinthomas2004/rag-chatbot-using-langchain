"""
Document loader module supporting PDF, TXT, and Markdown files.
Extracts page-by-page text content and cleans whitespace formatting.
"""

import re
from typing import List, Dict, Any
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Clean extra spaces and redundant newlines while preserving paragraph structure."""
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def load_pdf(file_path_or_buffer) -> List[Dict[str, Any]]:
    """
    Extract text and page metadata from a PDF file path or file-like object.
    
    Returns:
        List of dicts: [{"page": int, "text": str}]
    """
    reader = PdfReader(file_path_or_buffer)
    documents = []

    for page_idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)
        if cleaned:
            documents.append({
                "page": page_idx + 1,
                "text": cleaned
            })

    return documents


def load_text(file_path_or_buffer, default_name: str = "document") -> List[Dict[str, Any]]:
    """Extract and clean text from plain text or Markdown files."""
    if hasattr(file_path_or_buffer, "read"):
        content = file_path_or_buffer.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
    else:
        with open(file_path_or_buffer, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

    cleaned = clean_text(content)
    if not cleaned:
        return []

    return [{
        "page": 1,
        "text": cleaned
    }]
