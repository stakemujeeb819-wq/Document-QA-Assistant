from llama_index.llms.groq import Groq

from src.search import chunk_documents, build_embeddings, top_matches
from src.prompts import PROMPTS


class RAGPipeline:
    """Retrieve relevant chunks, build a prompt, and ask the Groq LLM."""

    def __init__(self, documents, prompt_version: str = "v3_refuse",
                 model_name: str = "openai/gpt-oss-20b", k: int = 3):
        self.chunks = chunk_documents(documents)
        self.model, self.vectors = build_embeddings(self.chunks)
        self.prompt_template = PROMPTS[prompt_version]
        self.llm = Groq(model=model_name)
        self.k = k

    def ask(self, question: str) -> dict:
        matches = top_matches(question, self.model, self.vectors, self.chunks, k=self.k)
        context = "\n\n".join(text for text, _ in matches)
        prompt = self.prompt_template.format(context=context, question=question)
        response = self.llm.complete(prompt)
        return {
            "question": question,
            "answer": str(response).strip(),
            "sources": [{"text": text, "score": round(score, 3)} for text, score in matches],
        }