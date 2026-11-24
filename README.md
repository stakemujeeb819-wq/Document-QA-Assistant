# Document Q&A Assistant (RAG)

A question-answering app for text and PDF documents. Upload files, ask questions, and get answers with the source passages they came from. Built in Python as a learning portfolio project covering retrieval-augmented generation (RAG), evaluation, data storage, and automation.

## Tech stack
Python 3.11, LlamaIndex, Groq API (LLM), Hugging Face embeddings (BAAI/bge-small-en-v1.5), NumPy, Pandas, Chroma (vector database), SQLite, Streamlit, pytest, Git

## How it works
1. **Load:** `src/document_store.py` reads `.txt` and `.pdf` files from a folder, with error handling for missing or unreadable files.
2. **Chunk:** `src/search.py` splits documents into overlapping pieces.
3. **Embed:** each chunk becomes a 384-number vector using a local embedding model.
4. **Retrieve:** the question is embedded and the top chunks are found, either by cosine similarity with NumPy (`src/search.py`) or by a persistent Chroma index (`src/vector_store.py`).
5. **Answer:** `src/rag.py` combines the chunks with a prompt template (`src/prompts.py`) and asks a Groq-hosted LLM.
6. **Show:** `app.py` (Streamlit) displays the answer and the source passages with scores.

## Features by stage

| Stage | What was built | Main files |
|---|---|---|
| 1 | Project structure, `.gitignore`, Git | `.gitignore` |
| 2 | Document loader class with error handling | `src/document_store.py` |
| 3 | Question cleaning with Pandas, statistics with NumPy | `src/evaluation.py` |
| 4 | Chunking, embeddings, cosine similarity | `src/search.py` |
| 5 | Three prompt templates | `src/prompts.py` |
| 6 | Full RAG pipeline with sources | `src/rag.py` |
| 7 | Streamlit app: upload, index, chat, sources | `app.py` |
| 8 | Keyword-based evaluation and runner | `src/run_eval.py`, `test_eval_run.py` |
| 8b | Prompt comparison and readable report | `test_prompt_compare.py`, `src/format_report.py` |
| 9 | SQLite storage and SQL analysis (JOIN, GROUP BY, subquery) | `src/db.py`, `sql/analysis.sql`, `run_analysis.py` |
| 10 | Chroma vector database backend | `src/vector_store.py`, `test_chroma.py` |
| 11 | LLM-as-judge evaluation with agreement check | `src/judge.py`, `test_judge.py` |
| 12 | Batch automation with logging and a Bash script | `scripts/run_batch.py`, `scripts/run_batch.sh` |
| 13 | pytest tests | `tests/` |
| 14 | Data warehouse: ETL to a star schema | `src/warehouse.py`, `run_warehouse.py`, `docs/warehouse.md` |

## Setup

```cmd
git clone https://github.com/stakemujeeb819-wq/Document-QA-Assistant.git
cd Document-QA-Assistant
uv venv --python 3.11
.venv\Scripts\activate
uv pip install llama-index-core llama-index-embeddings-huggingface llama-index-llms-groq llama-index-vector-stores-chroma chromadb streamlit pypdf pandas numpy pytest
set GROQ_API_KEY=your-groq-key
streamlit run app.py
```

Set `GROQ_API_KEY` in your own environment. Never commit a key.

## Running the tests

```cmd
python -m pytest -v
```

Four tests cover document loading, a chunking check, and the cosine ranking formula. The chunking test uses a simple splitter in place of LlamaIndex's sentence splitter, because NLTK's stopword files were blocked by a Windows file-permission check in the development environment.

## Evaluation

The test set is `tests_data/questions.csv`: 12 questions, of which 8 are answerable from `data/notes.txt` and 4 are not (the correct behavior is to say "I don't know").

| Method | Result |
|---|---|
| Prompt v1_basic (keyword check) | 75.0% (9/12) |
| Prompt v2_grounded (keyword check) | 66.7% (8/12) |
| Prompt v3_refuse (keyword check) | 100.0% (12/12) |
| v3_refuse, LLM judge | 91.7% on the first run; 100% after correcting one reference answer |

Limitations, stated plainly:
- 12 questions is a small, self-written test set.
- The keyword checker was refined during development, so the v3 result is not an independent measure.
- The judge uses the same model family as the system it grades, and one reference answer was corrected after the first run. Both judge runs are kept in the project history.

Results are saved in `tests_data/`: `prompt_comparison.csv`, `readable_report.txt`, `judge_results.csv`.

## Storage and analysis

Every evaluation run is saved to SQLite (`tests_data/app.db`, not committed). `sql/analysis.sql` contains three queries: a JOIN of questions with their source scores, a GROUP BY count per prompt, and a subquery for questions with below-average retrieval scores.

A star schema is built from those tables by `src/warehouse.py`: a `fact_answers` table (one row per question per prompt per run) with two dimension tables, `dim_question` and `dim_prompt`. The design is described in `docs/warehouse.md`.

## Automation

`scripts/run_batch.py` runs every question through the pipeline and the judge, logs each step to `logs/run.log`, and saves a timestamped CSV. Run it with:

```cmd
python -m scripts.run_batch
```

`scripts/run_batch.sh` does the same from Git Bash, and stops if the environment or the API key is missing.

## Project structure

```
app.py                    Streamlit UI
src/
  document_store.py       Loads documents
  search.py               Chunking and NumPy cosine search
  vector_store.py         Chroma vector store
  prompts.py              Prompt templates
  rag.py                  RAG pipeline
  evaluation.py           Loads and cleans test questions
  run_eval.py             Keyword evaluation runner
  judge.py                LLM-as-judge
  db.py                   SQLite storage
  warehouse.py            ETL and star schema
  format_report.py        Readable comparison report
scripts/                  Batch automation (Python and Bash)
sql/analysis.sql          Analysis queries
tests/                    pytest tests
tests_data/               Test questions and result files
docs/warehouse.md         Warehouse design
```

## Known limitations and next steps
- No authentication or multi-user support.
- Keyword and judge scores are proxies, not a full evaluation.
- Surrogate keys in the warehouse can change between runs.
- Planned: Claude API as an alternative LLM provider.