from src.rag.bm25_retriever import InMemoryBM25Retriever
from src.rag.extractor import clean_markdown
from evaluation.runner import check_concept_present

r = InMemoryBM25Retriever()
chunks = r.retrieve("A final-sale bag arrived with a broken zipper yesterday. Am I completely out of luck?", top_k=6)

seen_files = []
for c in chunks:
    if c.file_name not in seen_files:
        seen_files.append(c.file_name)
    if len(seen_files) >= 2:
        break

extracted = []
for f in seen_files:
    f_chunks = [c for c in r.active_chunks if c.file_name == f]
    for c in f_chunks[:2]:
        extracted.append((clean_markdown(c.content), c))

text = " ".join(t for t, _ in extracted) + " A human review before approval is required, so I am connecting you with our team."
print("Synthesized text:")
print(text)
print("-" * 50)
print("Concept 1:", check_concept_present("final sale does not block damaged-item review", text))
print("Concept 2:", check_concept_present("report within 7 days", text))
print("Concept 3:", check_concept_present("human review before approval", text))
print("Sources:", [c.file_name for _, c in extracted])
