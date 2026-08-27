import re
from src.rag.bm25_retriever import InMemoryBM25Retriever, tokenize, stem

cleaned = "can i change the shipping address on ord-1002 to my new apartment"
has_address = "address" in cleaned
print("has address:", has_address)
r = InMemoryBM25Retriever()
chunks = r.retrieve("Can I change the shipping address on ORD-1002 to my new apartment?", top_k=3)
print("Chunks for address change:")
for i, c in enumerate(chunks):
    print(f"[{i}] {c.file_name} > {c.heading} (score: {c.score:.2f})")
