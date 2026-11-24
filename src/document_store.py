from pathlib import Path
from llama_index.core import SimpleDirectoryReader


class DocumentStore:
    """Loads and tracks the documents in a data folder."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.documents = []

    def load(self):
        """Load all supported files from the data folder."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data folder not found: {self.data_dir}")

        try:
            reader = SimpleDirectoryReader(str(self.data_dir))
            self.documents = reader.load_data()
        except Exception as error:
            raise RuntimeError(f"Could not read documents: {error}") from error

        return self.documents

    def count(self) -> int:
        return len(self.documents)