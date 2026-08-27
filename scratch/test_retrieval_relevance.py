import math
from src.rag.bm25_retriever import InMemoryBM25Retriever

r = InMemoryBM25Retriever()

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
