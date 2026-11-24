from src.document_store import DocumentStore
from src.search import chunk_documents, build_embeddings, top_matches

store = DocumentStore("data")
docs = store.load()

chunks = chunk_documents(docs)
print(f"Created {len(chunks)} chunk(s)")

model, vectors = build_embeddings(chunks)
print(f"Vector matrix shape: {vectors.shape}")

for text, score in top_matches("What is LlamaIndex?", model, vectors, chunks):
    print(f"{score:.3f} | {text[:80]}")