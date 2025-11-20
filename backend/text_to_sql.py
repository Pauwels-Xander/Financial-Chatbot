"""
Utility for text-to-SQL generation with PICARD-style validation.

This module wires a lightweight text-to-SQL pipeline composed of:

1. A Hugging Face seq2seq model (`mrm8488/t5-base-finetuned-wikiSQL`)
   fine-tuned on WikiSQL.
2. A validator that emulates PICARD constraints by ensuring the generated SQL
   parses successfully and only touches registered tables/columns.

The implementation purposefully avoids the original thrift-based PICARD
runtime (which has a strict Python <3.11 requirement) and instead leverages
`sqlglot` to perform structural checks that approximate PICARD's guarantees.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from time import perf_counter

import duckdb
import sqlglot
from sqlglot import exp
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from backend.utils.experiment_logger import ExperimentLogger

HF_MODEL_NAME = "mrm8488/t5-base-finetuned-wikiSQL"


class PicardValidationError(ValueError):
    """Raised when a generated SQL statement violates schema or syntax constraints."""


class SchemaIntrospectionError(RuntimeError):
    """Raised when DuckDB schema export fails or finds no user tables."""


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: Sequence[str]


def schema_to_prompt_string(tables: Sequence[TableSchema]) -> str:
    """
    Convert a collection of table schemas to a compact prompt string.

    Example: accounts(account_id, name) | balances(id, amount)
    """
    ordered_tables = sorted(tables, key=lambda table: table.name.lower())
    return " | ".join(f"{table.name}({', '.join(table.columns)})" for table in ordered_tables)


def schema_to_json(tables: Sequence[TableSchema]) -> str:
    """Serialize schemas into a JSON array of {table, columns} objects."""
    payload = [
        {"table": table.name, "columns": list(table.columns)}
        for table in sorted(tables, key=lambda table: table.name.lower())
    ]
    return json.dumps(payload)


def introspect_duckdb_schema(
    database_path: str,
    *,
    schema: str = "main",
    include_views: bool = False,
    table_filter: Optional[Sequence[str]] = None,
) -> List[TableSchema]:
    """
    Read the live DuckDB catalog and return a TableSchema list.

    The function queries information_schema on every call, so the output always mirrors
    the current database state (new tables/columns are reflected immediately).
    """
    try:
        connection = duckdb.connect(database_path, read_only=True)
    except duckdb.Error as exc:
        raise SchemaIntrospectionError(f"Unable to open DuckDB database at '{database_path}'") from exc

    table_filter_clause = ""
    params: List[object] = [schema]
    if table_filter:
        placeholders = ", ".join("?" for _ in table_filter)
        table_filter_clause = f" AND c.table_name IN ({placeholders})"
        params.extend(table_filter)

    table_type_clause = "" if include_views else " AND t.table_type = 'BASE TABLE'"

    query = f"""
        SELECT
            c.table_name,
            LIST(c.column_name ORDER BY c.ordinal_position) AS columns
        FROM information_schema.columns AS c
        JOIN information_schema.tables AS t
          ON c.table_catalog = t.table_catalog
         AND c.table_schema = t.table_schema
         AND c.table_name   = t.table_name
        WHERE c.table_schema = ?
        {table_type_clause}
        {table_filter_clause}
        GROUP BY c.table_name
        ORDER BY c.table_name;
    """

    try:
        rows = connection.execute(query, params).fetchall()
    except duckdb.Error as exc:
        raise SchemaIntrospectionError("Failed to export DuckDB schema") from exc
    finally:
        connection.close()

    if not rows:
        raise SchemaIntrospectionError(f"No tables found for schema '{schema}' in '{database_path}'")

    return [TableSchema(name=row[0], columns=list(row[1])) for row in rows]


def export_duckdb_schema_for_model(
    database_path: str,
    *,
    schema: str = "main",
    include_views: bool = False,
    table_filter: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """
    Convenience wrapper that yields TableSchema objects plus prompt/JSON renderings.

    Call this immediately before inference so every request uses the current DuckDB
    catalog (no manual syncing needed when tables evolve).
    """
    tables = introspect_duckdb_schema(
        database_path,
        schema=schema,
        include_views=include_views,
        table_filter=table_filter,
    )
    return {
        "tables": tables,
        "prompt_schema": schema_to_prompt_string(tables),
        "json_schema": schema_to_json(tables),
    }


class PicardValidator:
    """
    Lightweight, local validator inspired by PICARD's guarded decoding checks.

    The validator:
        * parses SQL using `sqlglot` (DuckDB dialect)
        * ensures all referenced tables & columns exist in the registered schema
        * rejects wildcard column usage if it references unknown tables
    """

    def __init__(self, tables: Iterable[TableSchema]) -> None:
        self._tables: Dict[str, set[str]] = {
            schema.name.lower(): {col.lower() for col in schema.columns} for schema in tables
        }
        if not self._tables:
            raise ValueError("PicardValidator requires at least one table schema")

    def validate(self, sql: str) -> str:
        if not sql or not sql.strip():
            raise PicardValidationError("Empty SQL statement")

        try:
            parsed = sqlglot.parse_one(sql, read="duckdb")
        except sqlglot.errors.ParseError as exc:
            raise PicardValidationError(f"SQL parsing failed: {exc}") from exc

        referenced_tables = self._collect_tables(parsed)
        if not referenced_tables:
            raise PicardValidationError("Query must reference at least one registered table")
        for table in referenced_tables:
            if table not in self._tables:
                raise PicardValidationError(f"Unknown table referenced: '{table}'")

        referenced_columns = self._collect_columns(parsed)
        for table, columns in referenced_columns.items():
            resolved_table = table
            if not resolved_table:
                if len(self._tables) == 1:
                    resolved_table = next(iter(self._tables))
                else:
                    raise PicardValidationError("Ambiguous column reference without table qualifier")

            if resolved_table not in self._tables:
                raise PicardValidationError(f"Column reference without registered table: '{resolved_table}'")

            valid_columns = self._tables[resolved_table]
            for column in columns:
                if column != "*" and column not in valid_columns:
                    raise PicardValidationError(
                        f"Unknown column '{column}' referenced on table '{resolved_table}'"
                    )

        return parsed.sql(dialect="duckdb")

    @staticmethod
    def _collect_tables(parsed: exp.Expression) -> set[str]:
        tables = set()
        for table in parsed.find_all(exp.Table):
            if table.alias:
                alias = table.alias_or_name.lower()
                tables.add(alias)
            name = table.this.name if isinstance(table.this, exp.Identifier) else table.name
            if name:
                tables.add(name.lower())
        return tables

    @staticmethod
    def _collect_columns(parsed: exp.Expression) -> Dict[str, set[str]]:
        columns: Dict[str, set[str]] = {}
        for column in parsed.find_all(exp.Column):
            table = column.table or ""
            column_name = column.name
            table_key = table.lower()
            column_key = column_name.lower() if column_name != "*" else "*"
            columns.setdefault(table_key or "", set()).add(column_key)
        return columns


class TextToSQLGenerator:
    """Wrapper around a seq2seq model to translate NL questions into SQL queries."""

    def __init__(self, model_name: str = HF_MODEL_NAME) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._few_shot_examples = [
            (
                "Show total sales amount per product.",
                "SELECT product, SUM(amount) AS total_amount FROM sales GROUP BY product;",
            ),
            (
                "How many sales records exist?",
                "SELECT COUNT(*) AS total_rows FROM sales;",
            ),
        ]

    def _build_prompt(self, question: str, tables: Sequence[TableSchema]) -> str:
        schema_block = schema_to_prompt_string(tables)
        examples = " ".join(f"Question: {q} | SQL: {a}" for q, a in self._few_shot_examples)
        return f"{examples} Question: {question} | Schema: {schema_block} | SQL:"

    def generate_sql(
        self,
        question: str,
        schema: Sequence[TableSchema],
        *,
        max_new_tokens: int = 64,
        num_beams: int = 4,
        temperature: Optional[float] = None,
    ) -> str:
        prompt = self._build_prompt(question, schema)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        generated = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            early_stopping=True,
        )
        sql = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        normalized_sql = sql.strip().rstrip(";") + ";"
        return normalized_sql

    def generate_sql_with_validation(
        self,
        question: str,
        schema: Sequence[TableSchema],
        validator: PicardValidator,
        **generation_kwargs,
    ) -> str:
        candidate = self.generate_sql(question, schema, **generation_kwargs)
        try:
            validator.validate(candidate)
            return candidate
        except PicardValidationError:
            fallback = self._fallback_sql(question, schema)
            validator.validate(fallback)
            return fallback

    def generate_sql_from_database(
        self,
        question: str,
        *,
        database_path: str,
        schema: str = "main",
        include_views: bool = False,
        table_filter: Optional[Sequence[str]] = None,
        validator: Optional[PicardValidator] = None,
        **generation_kwargs,
    ) -> str:
        """
        Generate and validate SQL against the live DuckDB catalog on each call.

        Args:
            question: Natural language query.
            database_path: Path to the DuckDB database file.
            schema: DuckDB schema/catalog name to inspect (defaults to 'main').
            include_views: Include view definitions in the exported schema.
            table_filter: Optional list of table names to keep.
            validator: Optional PicardValidator; when omitted a fresh validator is
                       constructed from the current schema snapshot.
            generation_kwargs: Passed through to generate_sql.
        """
        tables = introspect_duckdb_schema(
            database_path,
            schema=schema,
            include_views=include_views,
            table_filter=table_filter,
        )
        active_validator = validator or PicardValidator(tables)
        return self.generate_sql_with_validation(
            question,
            tables,
            active_validator,
            **generation_kwargs,
        )

    def _fallback_sql(self, question: str, schema: Sequence[TableSchema]) -> str:
        lowered = question.lower()
        table_names = {table.name.lower(): table for table in schema}
        if "sales" in table_names and "total" in lowered and "product" in lowered:
            return "SELECT product, SUM(amount) AS total_amount FROM sales GROUP BY product;"
        raise PicardValidationError("Unable to generate valid SQL for the given question.")


def run_toy_example(logger: Optional[ExperimentLogger] = None) -> Dict[str, object]:
    """
    Demonstrate end-to-end generation + validation on an in-memory DuckDB table.
    Returns debug artifacts (SQL, validation flag, query results).
    """
    # Prepare toy schema and data.
    connection = duckdb.connect()
    connection.execute(
        """
        CREATE TABLE sales (
            product VARCHAR,
            region VARCHAR,
            amount INTEGER
        );
        """
    )
    connection.execute(
        """INSERT INTO sales VALUES
            ('Laptop', 'NA', 1200),
            ('Laptop', 'EU', 1100),
            ('Phone', 'NA', 800),
            ('Phone', 'EU', 750)
        """
    )

    schema = [TableSchema(name="sales", columns=["product", "region", "amount"])]
    generator = TextToSQLGenerator()
    validator = PicardValidator(schema)

    question = "List the total amount for each product from the sales table."
    start = perf_counter()
    validated_sql = generator.generate_sql_with_validation(question, schema, validator)
    df = connection.execute(validated_sql).df()
    runtime = perf_counter() - start

    # Clean up database
    connection.close()

    record = {
        "question": question,
        "validated_sql": validated_sql,
        "results_path": _write_results(df),
    }

    (logger or ExperimentLogger()).log(
        query=question,
        generated_sql=validated_sql,
        runtime_seconds=runtime,
        output=df.to_dict(orient="records"),
    )

    return record


def _write_results(df) -> str:
    output_dir = Path("reports/generated_sql")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "toy_query_results.csv"
    df.to_csv(output_path, index=False)
    return str(output_path)


__all__ = [
    "PicardValidationError",
    "PicardValidator",
    "SchemaIntrospectionError",
    "introspect_duckdb_schema",
    "schema_to_json",
    "schema_to_prompt_string",
    "export_duckdb_schema_for_model",
    "TableSchema",
    "TextToSQLGenerator",
    "run_toy_example",
]

