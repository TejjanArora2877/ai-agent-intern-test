"""Generic passage and section extraction utilities for RAG evidence."""

import re
from typing import List, Tuple, Optional
from src.models.schemas import RetrievedChunk


def clean_markdown(text: str) -> str:
    """Strip markdown formatting (bold, italics, headers, lists) for clean textual synthesis."""
    # Remove header markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    # Remove blockquote markers
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
    # Normalize hyphenated compound numerals (e.g. 45-calendar-day -> 45 calendar days)
    text = re.sub(r"(\d+)-calendar-day", r"\1 calendar days", text)
    # Convert list dashes to space
    text = re.sub(r"^\s*-\s+", "", text, flags=re.MULTILINE)
    # Normalize whitespace
    text = " ".join(text.split())
    return text


def split_sentences(text: str) -> List[str]:
    """Split text into sentences cleanly."""
    clean = clean_markdown(text)
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    return [s.strip() for s in sentences if s.strip()]


def extract_relevant_chunk_content(
    chunks: List[RetrievedChunk],
    max_chunks: int = 3,
    distinct_files: bool = True,
) -> List[Tuple[str, RetrievedChunk]]:
    """
    Extract clean, full section content from top-ranked chunks in relevance order.
    
    Args:
        chunks: List of retrieved chunks.
        max_chunks: Maximum number of chunks to return.
        distinct_files: If True, selects the top section per distinct file.
        
    Returns:
        List of (clean_section_text, source_chunk) tuples.
    """
    if not chunks:
        return []

    results: List[Tuple[str, RetrievedChunk]] = []
    seen_files = set()
    for chunk in chunks:
        if distinct_files and chunk.file_name in seen_files:
            continue
        seen_files.add(chunk.file_name)
        clean_text = clean_markdown(chunk.content)
        results.append((clean_text, chunk))
        if len(results) >= max_chunks:
            break
            
    return results


def extract_top_document_sections(
    chunks: List[RetrievedChunk],
    max_sections: int = 3,
) -> List[Tuple[str, RetrievedChunk]]:
    """Extract all retrieved sections belonging to the top-ranked document."""
    if not chunks:
        return []
    top_file = chunks[0].file_name
    same_doc_chunks = [c for c in chunks if c.file_name == top_file]
    return [(clean_markdown(c.content), c) for c in same_doc_chunks[:max_sections]]


def extract_multi_source_passages(
    chunks: List[RetrievedChunk],
    max_files: int = 2,
    sections_per_file: int = 2,
    all_active_chunks: Optional[List[RetrievedChunk]] = None,
) -> List[Tuple[str, RetrievedChunk]]:
    """Extract top sections across each distinct top-ranked document file."""
    if not chunks:
        return []
    
    seen_files: List[str] = []
    for c in chunks:
        if c.file_name not in seen_files:
            seen_files.append(c.file_name)
        if len(seen_files) >= max_files:
            break
            
    results: List[Tuple[str, RetrievedChunk]] = []
    pool = all_active_chunks if all_active_chunks is not None else chunks
    for f in seen_files:
        file_chunks = [c for c in pool if c.file_name == f]
        for c in file_chunks[:sections_per_file]:
            results.append((clean_markdown(c.content), c))
            
    return results


def query_has_coverage_in_chunks(query: str, chunks: List[RetrievedChunk], min_overlap_ratio: float = 0.4) -> bool:
    """
    Check if query terms have sufficient representation in retrieved chunks.
    Used as fallback when retriever does not implement is_query_covered.
    """
    stopwords = {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "a", "an", "the", "and", "or", "but", "if", "because", "as",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "can", "could", "should", "would", "may", "might",
        "must", "shall", "will", "i", "my", "you", "your", "it", "its", "we",
        "our", "they", "their", "this", "that", "these", "those", "all", "any",
        "for", "of", "in", "to", "on", "with", "at", "by", "from", "up", "about",
        "into", "over", "after", "please", "tell", "give", "know", "check", "long",
        "much", "take", "get", "here"
    }
    
    query_words = [w for w in re.findall(r"\b[a-z0-9]+\b", query.lower()) if w not in stopwords and len(w) > 2]
    if not query_words:
        return True

    all_chunk_text = " ".join(f"{c.title} {c.heading} {c.content}".lower() for c in chunks)
    
    found_count = 0
    for w in query_words:
        stem = w.rstrip("s")
        if w in all_chunk_text or (len(stem) > 2 and stem in all_chunk_text):
            found_count += 1

    return (found_count / len(query_words)) >= min_overlap_ratio
