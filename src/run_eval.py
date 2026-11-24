import pandas as pd

from src.document_store import DocumentStore
from src.evaluation import load_questions
from src.rag import RAGPipeline
from src.prompts import PROMPTS


def run_evaluation(data_dir: str, questions_path: str, prompt_version: str) -> pd.DataFrame:
    """Ask every test question and mark the answer as correct or wrong."""
    docs = DocumentStore(data_dir).load()
    rag = RAGPipeline(docs, prompt_version=prompt_version)
    questions = load_questions(questions_path)

    rows = []
    for _, row in questions.iterrows():
        result = rag.ask(row["question"])
        answer = result["answer"]
        keyword = row["expected_keyword"]
        if keyword:
            correct = keyword.lower() in answer.lower()
        else:
            correct = "don't know" in answer.lower()
        rows.append({
            "question": row["question"],
            "expected_keyword": keyword,
            "answer": answer,
            "correct": correct,
            "prompt_version": prompt_version,
        })
    return pd.DataFrame(rows)


def accuracy(results: pd.DataFrame) -> float:
    return float(results["correct"].mean() * 100)