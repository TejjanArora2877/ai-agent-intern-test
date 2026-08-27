import math
import re
from collections import Counter
from typing import List, Dict, Tuple
from src.models.schemas import RetrievedChunk
from src.rag.parser import KnowledgeBaseParser
from src.rag.bm25_retriever import tokenize, stem, STOPWORDS
from src.config import KNOWLEDGE_BASE_DIR

class EnhancedBM25FRetriever:
    def __init__(self):
        chunks = []
        for fpath in sorted(list(KNOWLEDGE_BASE_DIR.glob("*.md"))):
            chunks.extend(KnowledgeBaseParser.parse_file(fpath))
            
        self.active_chunks = [
            c for c in chunks
            if c.metadata.status == "active"
            and c.metadata.policy_authority == "official"
            and c.metadata.customer_answering is not False
            and not c.metadata.superseded_by
        ]
        
        self.w_title = 3.5
        self.w_heading = 4.5
        self.w_body = 1.0
        self.k1 = 1.2
        self.b = 0.75
        
        self.corpus_size = len(self.active_chunks)
        self.doc_freqs = Counter()
        self.weighted_lens = []
        self.chunk_weighted_tfs = []
        
        for c in self.active_chunks:
            t_tokens = tokenize(c.title, filter_stopwords=True)
            h_tokens = tokenize(f"{c.heading} {' '.join(c.heading_hierarchy)}", filter_stopwords=True)
            b_tokens = tokenize(c.content, filter_stopwords=True)
            
            t_counts = Counter(t_tokens)
            h_counts = Counter(h_tokens)
            b_counts = Counter(b_tokens)
            
            all_terms = set(t_counts.keys()) | set(h_counts.keys()) | set(b_counts.keys())
            weighted_tf = {}
            for t in all_terms:
                tf_val = (self.w_title * t_counts[t]) + (self.w_heading * h_counts[t]) + (self.w_body * b_counts[t])
                weighted_tf[t] = tf_val
                self.doc_freqs[t] += 1
                
            w_len = (self.w_title * len(t_tokens)) + (self.w_heading * len(h_tokens)) + (self.w_body * len(b_tokens))
            self.weighted_lens.append(w_len)
            self.chunk_weighted_tfs.append(weighted_tf)
            
        self.avg_w_len = sum(self.weighted_lens) / self.corpus_size if self.corpus_size > 0 else 1.0

    def retrieve(self, query: str, top_k: int = 6):
        query_tokens = tokenize(query, filter_stopwords=True)
        if not query_tokens:
            return []
            
        query_text = " ".join(tokenize(query, filter_stopwords=False))
        # Build query bigrams
        raw_words = tokenize(query, filter_stopwords=False)
        query_bigrams = [f"{raw_words[i]} {raw_words[i+1]}" for i in range(len(raw_words)-1)]
        
        scores = []
        for idx, chunk in enumerate(self.active_chunks):
            w_tf = self.chunk_weighted_tfs[idx]
            w_len = self.weighted_lens[idx]
            
            # Base BM25F score
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
            # If chunk title or heading contains the primary intent
            title_heading = f"{chunk.title} {chunk.heading}".lower()
            for q_term in query_tokens:
                if q_term in ["return", "ship", "warranti", "damag", "cancel", "adjust", "care", "membership"]:
                    if stem(q_term) in title_heading:
                        score += 4.0
                        
            scores.append((score, idx))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            c = self.active_chunks[idx].model_copy(deep=True)
            c.score = score
            results.append(c)
        return results

r = EnhancedBM25FRetriever()
queries = [
    "How long does a regular customer have to return an unused backpack?",
    "Do you ship internationally?",
    "Do all Aster & Row products have a lifetime warranty?",
    "Can I put the entire Breeze Tumbler in the dishwasher?"
]

for q in queries:
    chunks = r.retrieve(q, top_k=6)
    print(f"=== Query: {q} ===")
    for i, c in enumerate(chunks):
        print(f"  [{i}] {c.file_name} > {c.heading} (score: {c.score:.2f})")
    print()
