from pathlib import Path

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

CHROMA_DIR = Path("tests_data/chroma_db")
COLLECTION = "documents"


def build_chroma_index(chunks: list[str], model_name: str = "BAAI/bge-small-en-v1.5",
                       persist_dir: Path = CHROMA_DIR):
    """Store chunks in a persistent Chroma collection and return a LlamaIndex index."""
    embed_model = HuggingFaceEmbedding(model_name=model_name)

    client = chromadb.PersistentClient(path=str(persist_dir))
    # Rebuild the collection each time so it matches the current chunks
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    nodes = [TextNode(text=c) for c in chunks]
    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
    return index, embed_model


def chroma_top_matches(index, question: str, k: int = 3):
    """Return the k most similar chunks with their scores."""
    retriever = index.as_retriever(similarity_top_k=k)
    results = retriever.retrieve(question)
    return [(r.node.get_content(), float(r.score or 0.0)) for r in results]