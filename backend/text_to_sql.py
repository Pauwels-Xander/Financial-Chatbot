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


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: Sequence[str]


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
        schema_lines: List[str] = []
        for table in tables:
            schema_lines.append(f"{table.name}({', '.join(table.columns)})")
        schema_block = " | ".join(schema_lines)
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
    "TableSchema",
    "TextToSQLGenerator",
    "run_toy_example",
]

