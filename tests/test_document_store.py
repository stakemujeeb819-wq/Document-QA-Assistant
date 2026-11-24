import pytest

from src.document_store import DocumentStore


def test_missing_folder_raises(tmp_path):
    store = DocumentStore(str(tmp_path / "does_not_exist"))
    with pytest.raises(FileNotFoundError):
        store.load()


def test_loads_text_file(tmp_path):
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    store = DocumentStore(str(tmp_path))
    docs = store.load()
    assert store.count() == 1
    assert "hello world" in docs[0].text