from src.document_store import DocumentStore
from src.search import chunk_documents, build_embeddings, top_matches
from src.vector_store import build_chroma_index, chroma_top_matches

docs = DocumentStore("data").load()
chunks = chunk_documents(docs)

# NumPy search (our own code)
model, vectors = build_embeddings(chunks)
numpy_best = top_matches("What is LlamaIndex?", model, vectors, chunks, k=1)

# Chroma search (persistent vector database)
index, _ = build_chroma_index(chunks)
chroma_best = chroma_top_matches(index, "What is LlamaIndex?", k=1)

print("NumPy top chunk: ", numpy_best[0][0][:80])
print("Chroma top chunk:", chroma_best[0][0][:80])
print("Same top chunk:  ", numpy_best[0][0].strip() == chroma_best[0][0].strip())