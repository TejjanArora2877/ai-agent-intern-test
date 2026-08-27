import math
import re
from collections import Counter
from typing import List, Set

STOPWORDS: Set[str] = {
    "a", "about", "all", "an", "and", "any", "are", "as", "at", "be", "been",
    "being", "but", "by", "can", "could", "did", "do", "does", "for", "from",
    "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its",
    "may", "might", "must", "my", "of", "on", "or", "our", "shall",
    "should", "so", "some", "that", "the", "their", "them", "there", "these",
    "they", "this", "those", "to", "too", "up", "was", "we", "were", "what",
    "when", "where", "which", "who", "whom", "whose", "why", "will", "with",
    "would", "you", "your", "tell", "please", "check"
}

def stem(word: str) -> str:
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
    if filter_stopwords:
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens

from src.rag.parser import KnowledgeBaseParser
from src.config import KNOWLEDGE_BASE_DIR

chunks = []
for f in KNOWLEDGE_BASE_DIR.glob("*.md"):
    chunks.extend(KnowledgeBaseParser.parse_file(f))

active_chunks = [c for c in chunks if c.metadata.status == "active" and c.metadata.policy_authority == "official" and not c.metadata.superseded_by]

print(f"Total active chunks: {len(active_chunks)}")
q = "Can you ship an Atlas Weekender to Germany?"
q_tokens = tokenize(q, filter_stopwords=True)
print("Q tokens:", q_tokens)

for c in active_chunks:
    c_tokens = tokenize(f"{c.title} {c.heading} {c.content}")
    overlap = set(q_tokens).intersection(set(c_tokens))
    if overlap:
        print(f"{c.file_name} > {c.heading} -> Overlap: {overlap}")
