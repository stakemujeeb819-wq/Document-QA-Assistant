import sqlite3

import pandas as pd

from src.db import DB_PATH
from src.warehouse import run_etl

SUMMARY_SQL = """
SELECT p.prompt_version,
       COUNT(*)                        AS runs,
       ROUND(AVG(f.avg_score), 3)      AS avg_source_score,
       ROUND(AVG(f.answer_length), 1)  AS avg_words
FROM fact_answers f
JOIN dim_prompt p ON p.prompt_id = f.prompt_id
GROUP BY p.prompt_version
ORDER BY p.prompt_version;
"""

if __name__ == "__main__":
    fact = run_etl()
    print(f"Loaded fact_answers with {len(fact)} rows")

    conn = sqlite3.connect(DB_PATH)
    print(pd.read_sql_query(SUMMARY_SQL, conn).to_string(index=False))
    conn.close()