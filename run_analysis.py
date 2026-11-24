import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("tests_data/app.db")
SQL_PATH = Path("sql/analysis.sql")


def run_queries(db_path: Path = DB_PATH, sql_path: Path = SQL_PATH) -> None:
    """Run each query in analysis.sql and print the result as a table."""
    statements = [s.strip() for s in sql_path.read_text(encoding="utf-8").split(";") if s.strip()]

    conn = sqlite3.connect(db_path)
    try:
        for i, sql in enumerate(statements, start=1):
            # Drop comment lines so only the SQL is shown
            lines = [l for l in sql.splitlines() if not l.strip().startswith("--")]
            df = pd.read_sql_query("\n".join(lines), conn)
            print(f"\n--- Query {i} ({len(df)} rows) ---")
            print(df.to_string(index=False))
    finally:
        conn.close()


if __name__ == "__main__":
    run_queries()