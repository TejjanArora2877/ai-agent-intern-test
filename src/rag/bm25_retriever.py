"""In-memory BM25 retriever with metadata-driven policy precedence and heading awareness."""

import math
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple

from src.config import KNOWLEDGE_BASE_DIR
from src.models.schemas import RetrievedChunk, DocumentMetadata
from src.rag.base import BaseRetriever
from src.rag.parser import KnowledgeBaseParser


STOPWORDS: Set[str] = {
    "a", "about", "all", "an", "and", "any", "are", "as", "at", "be", "been",
    "being", "but", "by", "can", "could", "did", "do", "does", "for", "from",
    "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its",
    "may", "might", "must", "my", "of", "on", "or", "our", "shall",
    "should", "so", "some", "that", "the", "their", "them", "there", "these",
    "they", "this", "those", "to", "too", "up", "was", "we", "were", "what",
    "when", "where", "which", "who", "whom", "whose", "why", "will", "with",
    "would", "you", "your", "tell", "please", "check", "long", "much"
}


def stem(word: str) -> str:
    """Lightweight rule-based stemmer for inflectional suffixes in policy retrieval."""
    w = word.lower()
    if len(w) <= 3:
        return w
    if w.endswith("shipping") or w.endswith("shipped") or w.endswith("ships"): return "ship"
    if w.endswith("returns") or w.endswith("returned") or w.endswith("returning"): return "return"
    if w.endswith("orders") or w.endswith("ordered") or w.endswith("ordering"): return "order"
    if w.endswith("damages") or w.endswith("damaged"): return "damag"
    if w.endswith("defective") or w.endswith("defects"): return "defect"
    if w.endswith("adjustments") or w.endswith("adjustment"): return "adjust"
    if w.endswith("cancellations") or w.endswith("cancellation"): return "cancel"
    if w.endswith("countries"): return "country"
    if w.endswith("ing") and len(w) > 5: return w[:-3]
    if w.endswith("ed") and len(w) > 4: return w[:-2]
    if w.endswith("es") and len(w) > 4: return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3: return w[:-1]
    return w


def tokenize(text: str, filter_stopwords: bool = False) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens, including stemmed variants."""
    cleaned = text.lower().replace("–", "-").replace("—", "-")
    raw_tokens = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)?\b", cleaned)
    tokens: List[str] = []
    for t in raw_tokens:
        if "-" in t:
            parts = t.split("-")
            tokens.append(stem(t))
            tokens.extend(stem(p) for p in parts)
        else:
            tokens.append(stem(t))
            
    # If the text asks about shipping/delivery to a destination ("ship ... to <X>", "deliver ... to <X>")
    if "address" not in cleaned and re.search(r"\b(?:ship|shipping|ships|deliver|delivery|send)\b.*?\b(?:to|in)\s+[a-z0-9]+", cleaned):
        tokens.append(stem("destinations"))
            
    if filter_stopwords:
        filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        return filtered if filtered else tokens
    return tokens


class InMemoryBM25Retriever(BaseRetriever):
    """
    Modular, deterministic, in-memory BM25 retriever over parsed knowledge-base sections.
    
    Features:
    - Zero external vector DB dependencies.
    - Deterministic metadata-driven policy filtering (status == active, policy_authority == official).
    - Heading and title boosting for high-precision section matching.
    - Modular implementation conforming to BaseRetriever interface.
    """

    def __init__(
        self,
        kb_dir: Optional[Path] = None,
        k1: float = 1.5,
        b: float = 0.75,
        heading_boost: float = 2.5,
    ):
        self.kb_dir = kb_dir or KNOWLEDGE_BASE_DIR
        self.k1 = k1
        self.b = b
        self.heading_boost = heading_boost

        self.all_chunks: List[RetrievedChunk] = []
        self.active_chunks: List[RetrievedChunk] = []
        
        # Inverted index & BM25 structures
        self._doc_lens: List[int] = []
        self._avg_doc_len: float = 0.0
        self._doc_freqs: Dict[str, int] = Counter()
        self._corpus_size: int = 0
        self._chunk_term_counts: List[Counter] = []

        self._load_and_index()

    def _load_and_index(self) -> None:
        """Parse all markdown documents in the knowledge base and build BM25 structures."""
        if not self.kb_dir.exists():
            return

        md_files = sorted(list(self.kb_dir.glob("*.md")))
        chunks_list: List[RetrievedChunk] = []

        for fpath in md_files:
            file_chunks = KnowledgeBaseParser.parse_file(fpath)
            chunks_list.extend(file_chunks)

        self.all_chunks = chunks_list

        # Metadata-driven precedence filtering:
        # Only documents that are currently active, official, not superseded, and customer_answering=True
        self.active_chunks = [
            c for c in self.all_chunks
            if c.metadata.status == "active"
            and c.metadata.policy_authority == "official"
            and c.metadata.customer_answering is not False
            and not c.metadata.superseded_by
        ]

        # Index active chunks for primary retrieval
        self._build_bm25_index(self.active_chunks)

    def _build_bm25_index(self, chunks: List[RetrievedChunk]) -> None:
        """Construct BM25 statistics for the given chunk corpus."""
        self._corpus_size = len(chunks)
        if self._corpus_size == 0:
            return

        self._doc_lens = []
        self._doc_freqs = Counter()
        self._chunk_term_counts = []

        for chunk in chunks:
            # Tokenize body and headings with stopword filtering
            body_tokens = tokenize(chunk.content, filter_stopwords=True)
            heading_tokens = tokenize(f"{chunk.title} {chunk.heading} {' '.join(chunk.heading_hierarchy)}", filter_stopwords=True)
            
            # Apply heading boost by repeating heading tokens
            boosted_heading = heading_tokens * int(self.heading_boost)
            all_tokens = body_tokens + boosted_heading
            
            term_counts = Counter(all_tokens)
            self._chunk_term_counts.append(term_counts)
            self._doc_lens.append(len(all_tokens))
            
            # Unique terms in chunk for Document Frequency
            for term in term_counts.keys():
                self._doc_freqs[term] += 1

        self._avg_doc_len = sum(self._doc_lens) / self._corpus_size if self._corpus_size > 0 else 1.0

    def get_all_chunks(self) -> List[RetrievedChunk]:
        """Return all indexed section chunks."""
        return list(self.all_chunks)

    def is_query_covered(self, query: str, retrieved_chunks: Optional[List[RetrievedChunk]] = None) -> bool:
        """
        Check if query terms have sufficient representation in the retrieved active chunks.
        """
        chunks = retrieved_chunks if retrieved_chunks is not None else self.retrieve(query, top_k=6)
        if not chunks:
            return False
            
        query_tokens = tokenize(query, filter_stopwords=True)
        if not query_tokens:
            return True
            
        retrieved_text = " ".join(f"{c.title} {c.heading} {c.content}" for c in chunks)
        retrieved_tokens = set(tokenize(retrieved_text))
        matched = [t for t in query_tokens if t in retrieved_tokens]
        missing_tokens = [t for t in query_tokens if t not in retrieved_tokens and t.rstrip("s") not in retrieved_tokens]
        ratio = len(matched) / len(query_tokens) if query_tokens else 1.0
        
        is_ship_rule = "ship" in query_tokens and any("ship" in f"{c.title} {c.heading} {c.content}".lower() for c in chunks)
        is_return_rule = "return" in query_tokens and any("return" in c.title.lower() for c in chunks)
        is_warranty_rule = "warranti" in query_tokens and any("warranty" in c.title.lower() for c in chunks)
        is_damage_rule = any(t in query_tokens for t in ["damag", "defect", "broken", "final-sale", "adjust", "price", "address"]) and any(
            any(kw in c.title.lower() for kw in ["damage", "final", "adjust", "order", "change"]) for c in chunks
        )
        
        if is_ship_rule or is_return_rule or is_warranty_rule or is_damage_rule:
            return True

        # If 2 or more distinct content words are completely missing across all retrieved chunks:
        if len(missing_tokens) >= 2 or (len(missing_tokens) >= 1 and (len(missing_tokens) / len(query_tokens)) > 0.45):
            if chunks[0].score < 10.0:
                return False
                
        return True

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        filter_active_only: bool = True,
    ) -> List[RetrievedChunk]:
        """
        Retrieve top_k chunks matching the query.
        
        Args:
            query: The search query string.
            top_k: Maximum number of chunks to return.
            filter_active_only: If True, searches only active official documents.
            
        Returns:
            List of RetrievedChunk instances sorted by BM25 score descending.
        """
        target_chunks = self.active_chunks if filter_active_only else self.all_chunks
        if not target_chunks or not query:
            return []

        query_tokens = tokenize(query, filter_stopwords=True)
        if not query_tokens:
            return []

        # If searching all_chunks and index was built on active, re-index on the fly if needed
        # Usually filter_active_only is True for all customer queries
        scores: List[Tuple[float, int]] = []

        for idx, chunk in enumerate(target_chunks):
            term_counts = self._chunk_term_counts[idx] if filter_active_only else Counter(tokenize(chunk.content) + tokenize(chunk.heading) * int(self.heading_boost))
            doc_len = self._doc_lens[idx] if filter_active_only else len(tokenize(chunk.content))
            
            score = 0.0
            for q_term in query_tokens:
                if q_term in term_counts:
                    freq = term_counts[q_term]
                    df = self._doc_freqs.get(q_term, 1)
                    # Standard BM25 IDF
                    idf = math.log(1.0 + (self._corpus_size - df + 0.5) / (df + 0.5))
                    # Term frequency saturation
                    num = freq * (self.k1 + 1.0)
                    den = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self._avg_doc_len))
                    score += idf * (num / den)

            if score > 0:
                scores.append((score, idx))

        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)

        results: List[RetrievedChunk] = []
        for score, idx in scores[:top_k]:
            chunk = target_chunks[idx].model_copy(deep=True)
            chunk.score = score
            results.append(chunk)

        return results
