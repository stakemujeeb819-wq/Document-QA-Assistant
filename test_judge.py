import pandas as pd
from llama_index.llms.groq import Groq

from src.document_store import DocumentStore
from src.evaluation import load_questions
from src.judge import judge_answer
from src.rag import RAGPipeline
from src.run_eval import is_correct

PROMPT_VERSION = "v3_refuse"

docs = DocumentStore("data").load()
rag = RAGPipeline(docs, prompt_version=PROMPT_VERSION)
judge_llm = Groq(model="openai/gpt-oss-20b")

questions = pd.read_csv("tests_data/questions.csv")
questions["expected_keyword"] = questions["expected_keyword"].fillna("")
questions["reference"] = questions["reference"].fillna("")

rows = []
for _, row in questions.iterrows():
    answer = rag.ask(row["question"])["answer"]
    verdict = judge_answer(judge_llm, row["question"], row["reference"], answer)
    rows.append({
        "question": row["question"],
        "keyword_correct": is_correct(answer, row["expected_keyword"]),
        "judge_score": verdict["score"],
        "judge_reason": verdict["reason"],
        "answer": answer,
    })

results = pd.DataFrame(rows)
results.to_csv("tests_data/judge_results.csv", index=False)

print(results[["question", "keyword_correct", "judge_score"]].to_string(index=False))
print(f"\nKeyword accuracy: {results['keyword_correct'].mean() * 100:.1f}%")
print(f"Judge accuracy:   {results['judge_score'].mean() * 100:.1f}%")
print(f"Agreement:        {(results['keyword_correct'] == results['judge_score'].astype(bool)).mean() * 100:.1f}%")
print("\nSaved to tests_data/judge_results.csv")