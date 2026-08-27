from src.rag.bm25_retriever import InMemoryBM25Retriever
r = InMemoryBM25Retriever()
q = "Can you ship an Atlas Weekender to Germany?"
chunks = r.retrieve(q, top_k=6)
for i, c in enumerate(chunks):
    print(f"[{i}] {c.file_name} > {c.heading} (score: {c.score:.2f})")
