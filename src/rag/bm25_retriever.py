"""In-memory BM25F retriever with field weighting, coordination matching, and metadata-driven precedence."""

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
    "being", "but", "by", "can", "could", "did", "do", "does", "doe", "for", "from",
    "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its",
    "may", "might", "must", "my", "of", "on", "or", "our", "shall",
    "should", "so", "some", "that", "the", "their", "them", "there", "these",
    "they", "this", "those", "to", "too", "up", "was", "we", "were", "what",
    "when", "where", "which", "who", "whom", "whose", "why", "will", "with",
    "would", "you", "your", "tell", "please", "check", "long", "much",
    "take", "takes", "taking", "took", "get", "gets", "getting", "got", "give", "know"
}


def stem(word: str) -> str:
    """Lightweight rule-based stemmer for inflectional suffixes in policy retrieval."""
    w = word.lower()
    if len(w) <= 3:
        return w
    if w in ("broken", "defective", "defect", "defects", "flawed", "flaw"): return "defect"
    if w in ("damaged", "damages", "damage"): return "damag"
    if w.endswith("shipping") or w.endswith("shipped") or w.endswith("ships"): return "ship"
    if w.endswith("returns") or w.endswith("returned") or w.endswith("returning"): return "return"
    if w.endswith("orders") or w.endswith("ordered") or w.endswith("ordering"): return "order"
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
    Modular, deterministic, in-memory BM25F retriever over parsed knowledge-base sections.
    
    Features:
    - Zero external vector DB dependencies.
    - Deterministic metadata-driven policy filtering (status == active, policy_authority == official).
    - Field weighting (title=3.5, heading=4.5, body=1.0) for high-precision section matching.
    - Query coordination factor to heavily reward multi-term coverage over single incidental words.
    - Exact phrase / bigram proximity matching.
    - Domain alignment boost to prioritize relevant policy sections.
    - Modular implementation conforming to BaseRetriever interface.
    """

    def __init__(
        self,
        kb_dir: Optional[Path] = None,
        k1: float = 1.2,
        b: float = 0.75,
        w_title: float = 3.5,
        w_heading: float = 4.5,
        w_body: float = 1.0,
    ):
        self.kb_dir = kb_dir or KNOWLEDGE_BASE_DIR
        self.k1 = k1
        self.b = b
        self.w_title = w_title
        self.w_heading = w_heading
        self.w_body = w_body

        self.all_chunks: List[RetrievedChunk] = []
        self.active_chunks: List[RetrievedChunk] = []
        
        # BM25F structures
        self.doc_freqs: Dict[str, int] = Counter()
        self.weighted_lens: List[float] = []
        self.chunk_weighted_tfs: List[Dict[str, float]] = []
        self.corpus_size: int = 0
        self.avg_w_len: float = 0.0

        self._load_and_index()

    def _load_and_index(self) -> None:
        """Parse all markdown documents in the knowledge base and build BM25 structures."""
        if not self.kb_dir.exists():
            return

        chunks_list: List[RetrievedChunk] = []
        for fpath in sorted(list(self.kb_dir.glob("*.md"))):
            chunks_list.extend(KnowledgeBaseParser.parse_file(fpath))

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

        self.corpus_size = len(self.active_chunks)
        if self.corpus_size == 0:
            return

        self.doc_freqs = Counter()
        self.weighted_lens = []
        self.chunk_weighted_tfs = []

        for chunk in self.active_chunks:
            t_tokens = tokenize(chunk.title, filter_stopwords=True)
            h_tokens = tokenize(f"{chunk.heading} {' '.join(chunk.heading_hierarchy)}", filter_stopwords=True)
            b_tokens = tokenize(chunk.content, filter_stopwords=True)

            t_counts = Counter(t_tokens)
            h_counts = Counter(h_tokens)
            b_counts = Counter(b_tokens)

            all_terms = set(t_counts.keys()) | set(h_counts.keys()) | set(b_counts.keys())
            weighted_tf: Dict[str, float] = {}
            for t in all_terms:
                tf_val = (self.w_title * t_counts[t]) + (self.w_heading * h_counts[t]) + (self.w_body * b_counts[t])
                weighted_tf[t] = tf_val
                self.doc_freqs[t] += 1

            w_len = (self.w_title * len(t_tokens)) + (self.w_heading * len(h_tokens)) + (self.w_body * len(b_tokens))
            self.weighted_lens.append(w_len)
            self.chunk_weighted_tfs.append(weighted_tf)

        self.avg_w_len = sum(self.weighted_lens) / self.corpus_size if self.corpus_size > 0 else 1.0

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
            
        retrieved_text = " ".join(f"{c.title} {c.heading} {c.content}" for c in chunks).lower()
        retrieved_tokens = set(tokenize(retrieved_text))
        missing_tokens = [t for t in query_tokens if t not in retrieved_tokens and stem(t) not in retrieved_tokens]
        
        is_ship_rule = "ship" in query_tokens and any("ship" in f"{c.title} {c.heading} {c.content}".lower() for c in chunks)
        is_return_rule = "return" in query_tokens and any("return" in c.title.lower() for c in chunks)
        is_warranty_rule = "warranti" in query_tokens and any("warranty" in c.title.lower() for c in chunks)
        is_damage_rule = any(t in query_tokens for t in ["damag", "defect", "broken", "final-sale", "adjust", "price", "address"]) and any(
            any(kw in c.title.lower() for kw in ["damage", "final", "adjust", "order", "change"]) for c in chunks
        )
        
        if is_ship_rule or is_return_rule or is_warranty_rule or is_damage_rule:
            return True

        meaningful_missing = [t for t in missing_tokens if len(t) > 2 and t not in STOPWORDS]
        if len(meaningful_missing) >= 2 or (len(meaningful_missing) >= 1 and (len(meaningful_missing) / len(query_tokens)) >= 0.5):
            return False

        return True

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        filter_active_only: bool = True,
    ) -> List[RetrievedChunk]:
        """
        Retrieve top_k chunks matching the query using BM25F with coordination and field weighting.
        """
        target_chunks = self.active_chunks if filter_active_only else self.all_chunks
        if not target_chunks or not query:
            return []

        query_tokens = tokenize(query, filter_stopwords=True)
        if not query_tokens:
            return []

        raw_words = tokenize(query, filter_stopwords=False)
        query_bigrams = [f"{raw_words[i]} {raw_words[i+1]}" for i in range(len(raw_words)-1)]

        scores: List[Tuple[float, int]] = []

        for idx, chunk in enumerate(target_chunks):
            w_tf = self.chunk_weighted_tfs[idx] if filter_active_only else {}
            w_len = self.weighted_lens[idx] if filter_active_only else 1.0

            if not filter_active_only:
                t_tokens = tokenize(chunk.title, filter_stopwords=True)
                h_tokens = tokenize(f"{chunk.heading} {' '.join(chunk.heading_hierarchy)}", filter_stopwords=True)
                b_tokens = tokenize(chunk.content, filter_stopwords=True)
                t_counts = Counter(t_tokens)
                h_counts = Counter(h_tokens)
                b_counts = Counter(b_tokens)
                all_terms = set(t_counts.keys()) | set(h_counts.keys()) | set(b_counts.keys())
                w_tf = {t: (self.w_title * t_counts[t]) + (self.w_heading * h_counts[t]) + (self.w_body * b_counts[t]) for t in all_terms}
                w_len = (self.w_title * len(t_tokens)) + (self.w_heading * len(h_tokens)) + (self.w_body * len(b_tokens))

            score = 0.0
            matched_terms = 0
            for q_term in query_tokens:
                if q_term in w_tf:
                    matched_terms += 1
                    freq = w_tf[q_term]
                    df = self.doc_freqs.get(q_term, 1)
                    idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))
                    num = freq * (self.k1 + 1.0)
                    den = freq + self.k1 * (1.0 - self.b + self.b * (w_len / self.avg_w_len))
                    score += idf * (num / den)

            if score <= 0:
                continue

            # 1. Coordination Boost (percentage of query terms matched)
            coverage = matched_terms / len(query_tokens)
            coord_factor = 1.0 + (1.5 * (coverage ** 2))
            score *= coord_factor

            # 2. Bigram / Exact sequence match boost
            chunk_full_text = f"{chunk.title} {chunk.heading} {chunk.content}".lower()
            for bg in query_bigrams:
                if bg in chunk_full_text:
                    score += 3.0

            # 3. Domain Alignment Boost
            title_heading = f"{chunk.title} {chunk.heading}".lower()
            for q_term in query_tokens:
                st = stem(q_term)
                if st in ["return", "ship", "warranti", "damag", "defect", "broken", "cancel", "adjust", "care", "membership"]:
                    if st in title_heading or (st in ["defect", "broken"] and any(w in title_heading for w in ["damage", "defect", "wrong"])):
                        score += 4.0

            # 4. Companion Procedural & Reporting Window Boost
            # When a query addresses an issue (damaged, defective, broken, flaw, wrong item),
            # companion sections covering reporting windows, deadlines, and resolutions in that issue domain
            # must be prioritized alongside exception rules so the customer receives actionable timeframe guidance.
            is_issue_query = any(stem(w) in {"damag", "defect", "broken", "flaw", "wrong"} for w in query_tokens)
            if is_issue_query:
                is_reporting_section = any(term in title_heading for term in ["report", "window", "timeframe", "deadlin", "resolut", "claim", "except"])
                is_issue_domain = any(term in title_heading for term in ["damag", "defect", "wrong", "item", "warranti"])
                if is_reporting_section and is_issue_domain:
                    score += 6.0

            scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)

        results: List[RetrievedChunk] = []
        for score, idx in scores[:top_k]:
            chunk = target_chunks[idx].model_copy(deep=True)
            chunk.score = score
            results.append(chunk)

        return results
