from src.rag.bm25_retriever import InMemoryBM25Retriever
r = InMemoryBM25Retriever()
q = "I bought a final-sale Ridge Daypack 3 days ago and today the price dropped by $20. Can I get a price adjustment credited?"
chunks = r.retrieve(q, top_k=6)
for i, c in enumerate(chunks):
    print(f"[{i}] {c.file_name} > {c.heading} (score: {c.score:.2f})")
