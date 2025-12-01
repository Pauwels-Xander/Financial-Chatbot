import json
import time

import duckdb
import pandas as pd
import pytest

from backend.sql_executor import (
    DuckDBExecutor,
    SQLExecutionError,
    SQLExecutionTimeout,
)


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("duckdb") / "executor.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute("CREATE TABLE items (id INTEGER, name VARCHAR);")
    connection.execute(
        "INSERT INTO items VALUES (1, 'alpha'), (2, 'beta'), (3, 'gamma');"
    )
    connection.close()
    return str(path)


@pytest.fixture
def executor(db_path):
    executor = DuckDBExecutor(db_path, default_timeout=0.5)
    yield executor
    executor.close()


def test_run_returns_dataframe(executor):
    df = executor.run("SELECT * FROM items ORDER BY id")
    assert isinstance(df, pd.DataFrame)
    assert list(df["name"]) == ["alpha", "beta", "gamma"]


def test_run_can_return_json(executor):
    payload = executor.run(
        "SELECT id, name FROM items WHERE id = ?", params=(2,), as_json=True
    )
    records = json.loads(payload)
    assert records == [{"id": 2, "name": "beta"}]


def test_run_raises_sql_execution_error_on_failure(executor):
    with pytest.raises(SQLExecutionError):
        executor.run("SELECT * FROM missing_table")


def test_run_enforces_timeout(db_path, monkeypatch):
    slow_executor = DuckDBExecutor(db_path, default_timeout=0.05)
    original_execute = slow_executor._execute_query

    def slow_execute(query, params):
        time.sleep(0.2)
        return original_execute(query, params)

    monkeypatch.setattr(slow_executor, "_execute_query", slow_execute)

    with pytest.raises(SQLExecutionTimeout):
        slow_executor.run("SELECT * FROM items")

    slow_executor.close()


def test_run_invalid_sql_syntax_raises(executor):
    with pytest.raises(SQLExecutionError):
        executor.run("SELECT")

