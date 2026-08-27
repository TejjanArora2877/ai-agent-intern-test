import re
from src.rag.bm25_retriever import InMemoryBM25Retriever
from src.rag.extractor import clean_markdown, extract_top_document_sections

r = InMemoryBM25Retriever()

def test_query(q):
    chunks = r.retrieve(q, top_k=6)
    sections = extract_top_document_sections(chunks, max_sections=3)
    passages = " ".join(t for t, _ in sections)
    
    dest_match = re.search(r"\b(?:to|in)\s+([A-Z][a-z]+)\b", q)
    if dest_match and any("shipping" in c.title.lower() for _, c in sections):
        queried_dest = dest_match.group(1)
        if ("only to" in passages.lower() or "not available" in passages.lower()) and queried_dest.lower() not in passages.lower():
            passages = f"Shipping to {queried_dest} is not currently available. {passages}"
            
    print(f"Query: {q}")
    print(f"Passage: {passages[:120]}...")
    print(f"Sources: {[c.file_name for _, c in sections]}")
    print("-" * 50)

test_query("Can you ship an Atlas Weekender to Germany?")
test_query("What about Canada, and how long does it take?")
test_query("Can you ship to Australia?")
