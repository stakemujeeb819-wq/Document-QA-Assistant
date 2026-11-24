import sqlite3
from pathlib import Path

DB_PATH = Path("tests_data/app.db")


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the tables if they don't exist yet."""
    with get_connection(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                score REAL NOT NULL,
                FOREIGN KEY (interaction_id) REFERENCES interactions(id)
            );
        """)
    conn.close()


def save_interaction(question: str, answer: str, sources: list,
                     prompt_version: str, db_path: Path = DB_PATH) -> int:
    """Save one question, its answer, and its sources. Returns the interaction id."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO interactions (question, answer, prompt_version) VALUES (?, ?, ?)",
            (question, answer, prompt_version),
        )
        interaction_id = cur.lastrowid
        for s in sources:
            conn.execute(
                "INSERT INTO sources (interaction_id, chunk_text, score) VALUES (?, ?, ?)",
                (interaction_id, s["text"], s["score"]),
            )
    conn.close()
    return interaction_id


def fetch_all(db_path: Path = DB_PATH) -> list:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM interactions").fetchall()
    conn.close()
    return rows