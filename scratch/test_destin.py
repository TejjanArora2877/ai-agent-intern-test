import math
import re
from collections import Counter
from src.rag.bm25_retriever import InMemoryBM25Retriever, tokenize, stem
from src.rag.extractor import extract_top_document_sections

r = InMemoryBM25Retriever()

# Test retrieval for Germany, Canada, Australia, Japan
for q in [
    "Can you ship an Atlas Weekender to Germany?",
    "Do you ship to Australia?",
    "What about Canada, and how long does it take?",
    "Can you ship to Japan?"
]:
    # Custom retrieval test
    chunks = r.retrieve(q, top_k=4)
    print(f"Query: {q}")
    for i, c in enumerate(chunks):
        print(f"  [{i}] {c.file_name} > {c.heading} (score: {c.score:.2f})")
    print("-" * 50)
