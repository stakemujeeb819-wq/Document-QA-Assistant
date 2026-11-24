\# Warehouse design



\## Source tables (OLTP-style, written by the app)

\- interactions(id, question, answer, prompt\_version, created\_at)

\- sources(id, interaction\_id, chunk\_text, score)



\## Star schema (built by src/warehouse.py)



fact\_answers (grain: one row per question asked, per prompt, per run)

\- interaction\_id  (degenerate key to the source row)

\- question\_id     -> dim\_question

\- prompt\_id       -> dim\_prompt

\- source\_count, avg\_score, max\_score  (measures from retrieval)

\- answer\_length   (measure: words in the answer)

\- created\_at



dim\_question(question\_id, question)

dim\_prompt(prompt\_id, prompt\_version)



\## ETL steps

\- Extract: read both source tables from SQLite.

\- Transform: aggregate sources per interaction, compute answer length,

&#x20; assign surrogate keys to distinct questions and prompt versions.

\- Load: replace the three warehouse tables in the same database.



\## Limitation

Surrogate keys are assigned in order of appearance on each run, so

question\_id values can change between runs. A production warehouse

would keep keys stable with a lookup table.

