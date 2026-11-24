from src.document_store import DocumentStore
from src.rag import RAGPipeline

store = DocumentStore("data")
docs = store.load()

rag = RAGPipeline(docs, prompt_version="v3_refuse")
result = rag.ask("What is LlamaIndex?")

print("Answer:", result["answer"])
print("\nSources:")
for s in result["sources"]:
    print(f"  [{s['score']}] {s['text'][:80]}")