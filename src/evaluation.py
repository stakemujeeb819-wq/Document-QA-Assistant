import numpy as np
import pandas as pd


def load_questions(path: str) -> pd.DataFrame:
    """Load test questions from a CSV and clean them."""
    df = pd.read_csv(path)
    df = df.rename(columns=str.strip)
    df["question"] = df["question"].astype(str).str.strip()
    df["expected_keyword"] = df["expected_keyword"].fillna("").astype(str).str.strip()
    df = df[df["question"] != ""].drop_duplicates(subset="question")
    return df.reset_index(drop=True)


def question_stats(df: pd.DataFrame) -> dict:
    """Compute simple statistics about the question set with NumPy."""
    lengths = np.array([len(q.split()) for q in df["question"]])
    return {
        "total_questions": len(df),
        "avg_words": float(np.mean(lengths)),
        "longest_words": int(np.max(lengths)),
        "with_keyword": int((df["expected_keyword"] != "").sum()),
    }