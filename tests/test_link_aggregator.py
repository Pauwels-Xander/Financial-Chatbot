import json
from pathlib import Path

import pytest

import backend.embeddings.linker as linker_module
from backend.embeddings.linker import AccountLinker


class _DummyArray:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        return _DummyArray([[float(len(text))] * 3 for text in texts])


class DummyVectorDB:
    def __init__(self, directory=""):
        self.directory = Path(directory)

    def search(self, query_embedding, k=5):
        return [
            (
                999,
                0.1,
                {"account_id": 999, "text": "Revenue"},
            )
        ]

    def get_embedding_count(self):
        return 1

    def add_embeddings(self, ids, embeddings, metadata=None):
        pass


@pytest.fixture
def account_linker(monkeypatch, tmp_path):
    monkeypatch.setattr(linker_module, "SentenceTransformer", DummySentenceTransformer)
    monkeypatch.setattr(linker_module, "VectorDB", DummyVectorDB)

    lex_path = tmp_path / "lexicon.json"
    lex_path.write_text(json.dumps({"rev": "Revenue"}))

    return AccountLinker(
        model_name="dummy",
        vector_db_path=str(tmp_path / "vector"),
        account_lexicon_path=str(lex_path),
        duckdb_path=None,
    )


def test_link_accounts_returns_semantic_result(account_linker):
    results = account_linker.link_accounts("Revenue target")
    assert results
    assert results[0]["account_number"] == 999
    assert results[0]["match_type"] == "semantic"


def test_link_accounts_empty_query_returns_empty(account_linker):
    assert account_linker.link_accounts("") == []

