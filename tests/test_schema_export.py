import json

import duckdb
import pytest

from backend.text_to_sql import (
    SchemaIntrospectionError,
    export_duckdb_schema_for_model,
    introspect_duckdb_schema,
    schema_to_prompt_string,
)


def _build_db(tmp_path):
    db_path = tmp_path / "schema.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE accounts (account_id INTEGER, name VARCHAR);")
    con.execute("CREATE TABLE balances (id INTEGER, amount DECIMAL(18, 2));")
    con.close()
    return db_path


def test_export_schema_for_model_builds_prompt_and_json(tmp_path):
    db_path = _build_db(tmp_path)

    snapshot = export_duckdb_schema_for_model(str(db_path))
    prompt_schema = snapshot["prompt_schema"]
    json_schema = snapshot["json_schema"]

    assert "accounts(account_id, name)" in prompt_schema
    parsed = json.loads(json_schema)
    assert any(entry["table"] == "balances" for entry in parsed)
    assert any(entry["table"] == "accounts" for entry in parsed)


def test_schema_string_updates_when_schema_changes(tmp_path):
    db_path = _build_db(tmp_path)

    initial_tables = introspect_duckdb_schema(str(db_path), table_filter=["accounts"])
    accounts_table = next(item for item in initial_tables if item.name == "accounts")
    assert accounts_table.columns == ["account_id", "name"]
    assert "accounts(account_id, name)" == schema_to_prompt_string(initial_tables)

    con = duckdb.connect(str(db_path))
    con.execute("ALTER TABLE accounts ADD COLUMN region VARCHAR;")
    con.close()

    updated_tables = introspect_duckdb_schema(str(db_path), table_filter=["accounts"])
    updated_accounts = next(item for item in updated_tables if item.name == "accounts")
    assert "region" in updated_accounts.columns
    assert len(updated_accounts.columns) == 3
    assert "region" in schema_to_prompt_string(updated_tables)


def test_introspection_errors_on_empty_catalog(tmp_path):
    db_path = tmp_path / "empty.duckdb"
    con = duckdb.connect(str(db_path))
    con.close()

    with pytest.raises(SchemaIntrospectionError):
        introspect_duckdb_schema(str(db_path))
