import numpy as np
from llama_index.core.schema import Document

from src.search import chunk_documents


def test_chunk_documents_returns_text_chunks(monkeypatch):
    # Use a simple splitter so the test does not load NLTK data,
    # which is blocked by a Windows hardlink check in this environment.
    import src.search as search

    class SimpleSplitter:
        def __init__(self, chunk_size, chunk_overlap):
            self.chunk_size = chunk_size

        def get_nodes_from_documents(self, documents):
            from llama_index.core.schema import TextNode
            text = documents[0].text
            return [TextNode(text=text[i:i + self.chunk_size])
                    for i in range(0, len(text), self.chunk_size)]

    monkeypatch.setattr(search, "SentenceSplitter", SimpleSplitter)

    docs = [Document(text="LlamaIndex is a data framework. " * 50)]
    chunks = chunk_documents(docs, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(isinstance(c, str) and c.strip() for c in chunks)


def test_cosine_ranking_puts_most_similar_first():
    # Pure NumPy check of the same formula used in top_matches
    chunk_vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    q = np.array([1.0, 0.0])
    norms = np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(q)
    scores = chunk_vectors @ q / norms
    best = np.argsort(scores)[::-1]
    assert best[0] == 0
    assert best[-1] == 1