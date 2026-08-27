from src.rag.bm25_retriever import InMemoryBM25Retriever
r = InMemoryBM25Retriever()
chunks = r.retrieve("A final-sale bag arrived with a broken zipper yesterday. Am I completely out of luck?", top_k=15)
for i, c in enumerate(chunks):
    print(f"[{i}] {c.file_name} > {c.heading} (score: {c.score:.2f})")
