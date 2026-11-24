import numpy as np
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def chunk_documents(documents, chunk_size: int = 500, overlap: int = 50):
    """Split documents into smaller text chunks."""
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    return [node.get_content() for node in nodes]


def build_embeddings(chunks, model_name: str = "BAAI/bge-small-en-v1.5"):
    """Turn each chunk into a vector and stack them into a matrix."""
    model = HuggingFaceEmbedding(model_name=model_name)
    vectors = [model.get_text_embedding(c) for c in chunks]
    return model, np.array(vectors)


def top_matches(query: str, model, chunk_vectors, chunks, k: int = 3):
    """Return the k chunks most similar to the query, with cosine scores."""
    q = np.array(model.get_query_embedding(query))
    norms = np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(q)
    scores = chunk_vectors @ q / norms
    best = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in best]