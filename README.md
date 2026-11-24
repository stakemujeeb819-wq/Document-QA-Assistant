\# Document Q\&A Assistant (RAG)



A question-answering app that lets you upload text or PDF documents, then asks questions and returns answers with the source passages they came from.



\## Tech stack

Python, LlamaIndex, Groq API (LLM), Hugging Face embeddings (BAAI/bge-small-en-v1.5), NumPy, Pandas, Streamlit, Git



\## How it works

1\. \*\*Load:\*\* `src/document\_store.py` reads `.txt` and `.pdf` files from a folder.

2\. \*\*Chunk:\*\* `src/search.py` splits documents into overlapping pieces.

3\. \*\*Embed:\*\* each chunk becomes a 384-number vector using a local embedding model.

4\. \*\*Retrieve:\*\* the question is embedded, and cosine similarity (computed with NumPy) finds the top matching chunks.

5\. \*\*Answer:\*\* `src/rag.py` builds a prompt from the chunks (`src/prompts.py`) and asks a Groq-hosted LLM.

6\. \*\*Show:\*\* the Streamlit app in `app.py` displays the answer and the source passages with scores.



\## Setup



```powershell

git clone <your-repo-url>

cd Document\_QA\_Assistant

uv venv --python 3.11

.venv\\Scripts\\activate

uv pip install llama-index-core llama-index-embeddings-huggingface llama-index-llms-groq streamlit pypdf pandas numpy

$env:GROQ\_API\_KEY="your-groq-key"

streamlit run app.py

```



\## Evaluation



`test\_eval\_run.py` runs 12 self-written questions (answerable and unanswerable) through the pipeline and checks each answer against expected keywords.



| Prompt version | Accuracy |

|---|---|

| v1\_basic | 75.0% (9/12) |

| v2\_grounded | 66.7% (8/12) |

| v3\_refuse | 100.0% (12/12) |



Limitations: the test set is small, and the keyword checker was refined during development. These numbers are not a general accuracy measure.



\## Project structure

```

app.py              Streamlit UI

src/document\_store.py   Loads documents

src/search.py           Chunking and cosine similarity

src/prompts.py          Prompt templates

src/rag.py              Full RAG pipeline

src/evaluation.py       Loads and cleans test questions

src/run\_eval.py         Evaluation runner

src/format\_report.py    Readable report

tests\_data/             Test questions and results

```

