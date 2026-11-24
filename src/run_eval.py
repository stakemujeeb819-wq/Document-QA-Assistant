import pandas as pd

from src.db import init_db, save_interaction
from src.document_store import DocumentStore
from src.evaluation import load_questions
from src.rag import RAGPipeline


def is_correct(answer: str, keyword_field: str) -> bool:
    """Check the answer against one or more acceptable keywords.

    keyword_field examples:
      "framework"       -> answer must contain "framework"
      "APIs|PDFs|SQL"   -> answer must contain any one of them
      ""                -> answer must say it doesn't know
    """
    answer_lower = answer.lower()
    if keyword_field:
        options = [k.strip().lower() for k in keyword_field.split("|") if k.strip()]
        return any(k in answer_lower for k in options)
    return "don't know" in answer_lower or "do not know" in answer_lower


def run_evaluation(data_dir: str, questions_path: str, prompt_version: str) -> pd.DataFrame:
    """Ask every test question, save each run to SQLite, and mark the answer."""
    init_db()

    docs = DocumentStore(data_dir).load()
    rag = RAGPipeline(docs, prompt_version=prompt_version)
    questions = load_questions(questions_path)

    rows = []
    for _, row in questions.iterrows():
        result = rag.ask(row["question"])
        answer = result["answer"]
        keyword = row["expected_keyword"]

        save_interaction(row["question"], answer, result["sources"], prompt_version)

        rows.append({
            "question": row["question"],
            "expected_keyword": keyword,
            "answer": answer,
            "correct": is_correct(answer, keyword),
            "prompt_version": prompt_version,
        })
    return pd.DataFrame(rows)


def accuracy(results: pd.DataFrame) -> float:
    return float(results["correct"].mean() * 100)