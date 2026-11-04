"""
Text splitter module implementing character and recursive chunking with configurable overlap.
"""

from typing import List, Dict, Any


def character_chunking(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Splits text into chunks of given size with overlap using character indexing.
    Preserves exact sliding window chunks matching reference implementation.
    """
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be strictly less than chunk_size")

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap
    return chunks


def split_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 100,
    source_name: str = "document"
) -> List[Dict[str, Any]]:
    """
    Takes loaded page documents and produces tagged chunks with source metadata.
    
    Returns:
        List of dicts: [
            {"page": int, "chunk": str, "source": str, "chunk_id": int}
        ]
    """
    all_chunks = []
    global_chunk_id = 0

    for doc in documents:
        page = doc.get("page", 1)
        text = doc.get("text", "")
        page_chunks = character_chunking(text, chunk_size=chunk_size, overlap=overlap)

        for c in page_chunks:
            all_chunks.append({
                "page": page,
                "chunk": c,
                "source": source_name,
                "chunk_id": global_chunk_id
            })
            global_chunk_id += 1

    return all_chunks
