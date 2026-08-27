from src.rag.bm25_retriever import InMemoryBM25Retriever, tokenize
r = InMemoryBM25Retriever()
tokens = tokenize("Are all fabrics and adhesives in your bags vegan?", filter_stopwords=True)
corpus_vocab = set(r._doc_freqs.keys())
for t in tokens:
    stem_t = t.rstrip("s")
    direct = t in corpus_vocab
    plus_s = (t + "s") in corpus_vocab
    stem_in = len(stem_t) > 2 and stem_t in corpus_vocab
    sub = [w for w in corpus_vocab if len(w) > 3 and len(t) > 3 and (t in w or w in t)]
    print(f"Token '{t}': direct={direct}, plus_s={plus_s}, stem_in={stem_in}, sub={sub}")
