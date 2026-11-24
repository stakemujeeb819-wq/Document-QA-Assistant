import sqlite3

import pandas as pd

from src.db import DB_PATH


def extract(db_path=DB_PATH):
    """E: read the raw tables from SQLite."""
    conn = sqlite3.connect(db_path)
    try:
        interactions = pd.read_sql_query("SELECT * FROM interactions", conn)
        sources = pd.read_sql_query("SELECT * FROM sources", conn)
    finally:
        conn.close()
    return interactions, sources


def transform(interactions: pd.DataFrame, sources: pd.DataFrame):
    """T: build one fact table and two dimension tables with Pandas."""
    # Aggregate the source rows to one row per interaction
    src_stats = (
        sources.groupby("interaction_id")["score"]
        .agg(source_count="count", avg_score="mean", max_score="max")
        .reset_index()
    )
    df = interactions.merge(src_stats, left_on="id", right_on="interaction_id", how="left")
    df = df.drop(columns=["interaction_id"])  # avoid a duplicate column name
    df["source_count"] = df["source_count"].fillna(0).astype(int)
    df["answer_length"] = df["answer"].str.split().str.len()

    # Dimension tables: one row per distinct value, with a surrogate key
    dim_question = pd.DataFrame({"question": df["question"].unique()})
    dim_question["question_id"] = range(1, len(dim_question) + 1)

    dim_prompt = pd.DataFrame({"prompt_version": df["prompt_version"].unique()})
    dim_prompt["prompt_id"] = range(1, len(dim_prompt) + 1)

    # Fact table: keys to the dimensions plus the measures
    fact = (
        df.merge(dim_question, on="question")
        .merge(dim_prompt, on="prompt_version")
        .rename(columns={"id": "interaction_id"})
        [["interaction_id", "question_id", "prompt_id", "source_count",
          "avg_score", "max_score", "answer_length", "created_at"]]
    )
    return fact, dim_question[["question_id", "question"]], dim_prompt[["prompt_id", "prompt_version"]]


def load(fact, dim_question, dim_prompt, db_path=DB_PATH) -> None:
    """L: write the star schema back to SQLite as separate tables."""
    conn = sqlite3.connect(db_path)
    try:
        fact.to_sql("fact_answers", conn, if_exists="replace", index=False)
        dim_question.to_sql("dim_question", conn, if_exists="replace", index=False)
        dim_prompt.to_sql("dim_prompt", conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()


def run_etl(db_path=DB_PATH) -> pd.DataFrame:
    interactions, sources = extract(db_path)
    fact, dim_q, dim_p = transform(interactions, sources)
    load(fact, dim_q, dim_p, db_path)
    return fact