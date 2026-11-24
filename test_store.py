from src.document_store import DocumentStore

store = DocumentStore("data")
docs = store.load()
print(f"Loaded {store.count()} document(s)")