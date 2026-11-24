import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.document_store import DocumentStore
from src.evaluation import load_questions
from src.judge import judge_answer
from src.rag import RAGPipeline
from src.run_eval import is_correct
from llama_index.llms.groq import Groq

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
OUT_DIR = Path("tests_data/batch_runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "run.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("batch")


def run_batch(data_dir: str = "data", questions_path: str = "tests_data/questions.csv",
              prompt_version: str = "v3_refuse") -> Path:
    """Run every question through the pipeline and the judge, then save a timestamped CSV."""
    log.info("Batch started: prompt=%s, questions=%s", prompt_version, questions_path)

    try:
        docs = DocumentStore(data_dir).load()
        log.info("Loaded %d document(s) from %s", len(docs), data_dir)

        rag = RAGPipeline(docs, prompt_version=prompt_version)
        judge_llm = Groq(model="openai/gpt-oss-20b")

        questions = load_questions(questions_path)
        raw = pd.read_csv(questions_path).fillna("")
        references = dict(zip(raw["question"].str.strip(), raw["reference"]))
    except Exception as error:
        log.exception("Setup failed: %s", error)
        raise

    rows = []
    for _, row in questions.iterrows():
        q = row["question"]
        try:
            answer = rag.ask(q)["answer"]
            verdict = judge_answer(judge_llm, q, references.get(q, ""), answer)
            rows.append({
                "question": q,
                "answer": answer,
                "keyword_correct": is_correct(answer, row["expected_keyword"]),
                "judge_score": verdict["score"],
                "judge_reason": verdict["reason"],
            })
            log.info("OK | judge=%d | %s", verdict["score"], q)
        except Exception as error:
            log.error("FAILED | %s | %s", q, error)

    results = pd.DataFrame(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"batch_{prompt_version}_{stamp}.csv"
    results.to_csv(out_path, index=False)
    log.info("Batch finished: %d/%d succeeded, saved %s", len(rows), len(questions), out_path)
    return out_path


if __name__ == "__main__":
    path = run_batch()
    print(f"Saved results to {path}")
    print(f"Log file: {LOG_DIR / 'run.log'}")