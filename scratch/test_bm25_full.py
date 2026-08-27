import math
from collections import Counter
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from src.config import KNOWLEDGE_BASE_DIR
from src.models.schemas import RetrievedChunk
from src.rag.base import BaseRetriever
from src.rag.parser import KnowledgeBaseParser
from src.rag.bm25_retriever import tokenize, stem, STOPWORDS


class InMemoryBM25Retriever(BaseRetriever):
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
        
        self.doc_freqs: Dict[str, int] = Counter()
        self.weighted_lens: List[float] = []
        self.chunk_weighted_tfs: List[Dict[str, float]] = []
        self.corpus_size: int = 0
        self.avg_w_len: float = 0.0

        self._load_and_index()

    def _load_and_index(self) -> None:
        if not self.kb_dir.exists():
            return

        chunks_list: List[RetrievedChunk] = []
        for fpath in sorted(list(self.kb_dir.glob("*.md"))):
            chunks_list.extend(KnowledgeBaseParser.parse_file(fpath))

        self.all_chunks = chunks_list
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
        return list(self.all_chunks)

    def is_query_covered(self, query: str, retrieved_chunks: Optional[List[RetrievedChunk]] = None) -> bool:
        chunks = retrieved_chunks if retrieved_chunks is not None else self.retrieve(query, top_k=6)
        if not chunks:
            return False
            
        query_tokens = tokenize(query, filter_stopwords=True)
        if not query_tokens:
            return True
            
        retrieved_text = " ".join(f"{c.title} {c.heading} {c.content}" for c in chunks)
        retrieved_tokens = set(tokenize(retrieved_text))
        missing_tokens = [t for t in query_tokens if t not in retrieved_tokens and t.rstrip("s") not in retrieved_tokens]
        
        is_ship_rule = "ship" in query_tokens and any("ship" in f"{c.title} {c.heading} {c.content}".lower() for c in chunks)
        is_return_rule = "return" in query_tokens and any("return" in c.title.lower() for c in chunks)
        is_warranty_rule = "warranti" in query_tokens and any("warranty" in c.title.lower() for c in chunks)
        is_damage_rule = any(t in query_tokens for t in ["damag", "defect", "broken", "final-sale", "adjust", "price", "address"]) and any(
            any(kw in c.title.lower() for kw in ["damage", "final", "adjust", "order", "change"]) for c in chunks
        )
        
        if is_ship_rule or is_return_rule or is_warranty_rule or is_damage_rule:
            return True

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
                if q_term in ["return", "ship", "warranti", "damag", "cancel", "adjust", "care", "membership"]:
                    if stem(q_term) in title_heading:
                        score += 4.0

            scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)

        results: List[RetrievedChunk] = []
        for score, idx in scores[:top_k]:
            chunk = target_chunks[idx].model_copy(deep=True)
            chunk.score = score
            results.append(chunk)

        return results

r = InMemoryBM25Retriever()

# Test 1: Return-window question ranks return policy above warranty and shipping
res1 = r.retrieve("How long does a regular customer have to return an unused backpack?", top_k=6)
print("Query 1:", [(c.file_name, c.heading, round(c.score, 2)) for c in res1])

# Test 2: Shipping question ranks shipping sections above warranty sections
res2 = r.retrieve("If my order subtotal is $60 and I have a standard account, do I get free shipping in the US?", top_k=6)
print("Query 2:", [(c.file_name, c.heading, round(c.score, 2)) for c in res2])

# Test 3: Order-only question returns no irrelevant chunks
res3 = r.retrieve("Where is ORD-1007?", top_k=6)
print("Query 3 (Order only):", [(c.file_name, c.heading, round(c.score, 2)) for c in res3])
