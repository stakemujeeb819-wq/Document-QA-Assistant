import json
import re

from llama_index.llms.groq import Groq

JUDGE_PROMPT = """You are grading an answer from a document Q&A system.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Score 1 if the system answer is factually consistent with the reference answer and does not add unsupported claims.
Score 0 otherwise.
If the reference says the answer is unknown, score 1 only if the system says it does not know.

Respond with JSON only, no other text, in this format:
{{"score": 0 or 1, "reason": "one short sentence"}}"""


def judge_answer(llm: Groq, question: str, reference: str, answer: str) -> dict:
    """Ask the LLM to grade one answer. Returns {'score': int, 'reason': str}."""
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
    raw = str(llm.complete(prompt)).strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"score": 0, "reason": f"Unparseable judge output: {raw[:100]}"}
    try:
        data = json.loads(match.group(0))
        score = 1 if int(data.get("score", 0)) == 1 else 0
        return {"score": score, "reason": str(data.get("reason", ""))}
    except (json.JSONDecodeError, ValueError):
        return {"score": 0, "reason": f"Invalid JSON from judge: {raw[:100]}"}