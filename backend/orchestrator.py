"""
Pipeline orchestrator that integrates all components for end-to-end query processing.

This module orchestrates the complete pipeline:
1. Receive query
2. Time parser → extract temporal expressions
3. Topic classifier → categorize query
4. Entity linker → find relevant accounts
5. Text-to-SQL → generate SQL with PICARD validation
6. SQL executor → run query on DuckDB
7. LLM answer generator → format response
8. Experiment logger → record for debugging/evaluation

Returns structured JSON with intermediate outputs for debugging.
"""

from __future__ import annotations

import json
import re
import copy
import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import date
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

# Ensure project root is on sys.path when running this file directly
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from backend.embeddings.linker import AccountLinker
except ImportError:
    AccountLinker = None  # type: ignore
from backend.sql_executor import DuckDBExecutor, SQLExecutionError, SQLExecutionTimeout
from backend.text_to_sql import (
    TextToSQLGenerator,
    PicardValidator,
    PicardValidationError,
    introspect_duckdb_schema,
    TableSchema,
)
from backend.utils.answer_generator import AnswerGenerator
from backend.utils.experiment_logger import ExperimentLogger
from backend.utils.query_router import QueryRouter, QueryClassification
from backend.utils.time_parser import parse_time_expression, TimeParseError, TimeParseResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Structured result from the complete pipeline."""

    # Input
    query: str
    database_path: str

    # Intermediate outputs (for debugging)
    time_parse_result: Optional[Dict[str, Any]] = None
    query_classification: Optional[Dict[str, Any]] = None
    entity_links: Optional[List[Dict[str, Any]]] = None
    generated_sql: Optional[str] = None
    validation_status: Optional[str] = None
    sql_execution_result: Optional[Dict[str, Any]] = None

    # Final output
    answer: Optional[str] = None

    # Metadata
    runtime_seconds: float = 0.0
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PipelineOrchestrator:
    """
    Main orchestrator that coordinates all pipeline components.

    Processes natural language queries through the complete pipeline and
    returns structured results with intermediate outputs for debugging.
    """

    def __init__(
        self,
        database_path: str,
        *,
        entity_linker: Optional[Any] = None,  # AccountLinker if available
        sql_executor: Optional[DuckDBExecutor] = None,
        text_to_sql_generator: Optional[TextToSQLGenerator] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        query_router: Optional[QueryRouter] = None,
        experiment_logger: Optional[ExperimentLogger] = None,
        base_date: Optional[date] = None,
    ):
        """
        Initialize the pipeline orchestrator.

        Args:
            database_path: Path to the DuckDB database file
            entity_linker: Optional AccountLinker instance (will create if None)
            sql_executor: Optional DuckDBExecutor instance (will create if None)
            text_to_sql_generator: Optional TextToSQLGenerator instance (will create if None)
            answer_generator: Optional AnswerGenerator instance (will create if None)
            query_router: Optional QueryRouter instance (will create if None)
            experiment_logger: Optional ExperimentLogger instance (will create if None)
            base_date: Optional base date for time parsing (defaults to today)
        """
        self.database_path = database_path
        self.base_date = base_date or date.today()

        # Initialize components (lazy loading for optional components)
        self._entity_linker = entity_linker
        self._sql_executor = sql_executor
        self._text_to_sql_generator = text_to_sql_generator
        self._answer_generator = answer_generator or AnswerGenerator()
        self._query_router = query_router or QueryRouter()
        self._experiment_logger = experiment_logger or ExperimentLogger()
        self._cache_enabled = True
        self._cached_compute = lru_cache(maxsize=32)(self._compute_result_uncached)

    @property
    def entity_linker(self) -> Optional[AccountLinker]:
        """Lazy-load entity linker if needed."""
        if AccountLinker is None:
            return None
        if self._entity_linker is None:
            self._entity_linker = AccountLinker(duckdb_path=self.database_path)
        return self._entity_linker

    @property
    def sql_executor(self) -> DuckDBExecutor:
        """Lazy-load SQL executor if needed."""
        if self._sql_executor is None:
            self._sql_executor = DuckDBExecutor(self.database_path, default_timeout=10.0)
        return self._sql_executor

    @property
    def text_to_sql_generator(self) -> TextToSQLGenerator:
        """Lazy-load text-to-SQL generator if needed."""
        if self._text_to_sql_generator is None:
            self._text_to_sql_generator = TextToSQLGenerator()
        return self._text_to_sql_generator

    def process_query(self, query: str, *, log_experiment: bool = True) -> PipelineResult:
        """
        Process a natural language query through the complete pipeline.

        Args:
            query: Natural language question
            log_experiment: Whether to log this experiment for debugging/evaluation

        Returns:
            PipelineResult with all intermediate outputs and final answer
        """
        start_time = perf_counter()
        compute_fn = self._cached_compute if self._cache_enabled else self._compute_result_uncached
        cache_info_before = compute_fn.cache_info() if self._cache_enabled else None
        base_result = compute_fn(query)
        cache_info_after = compute_fn.cache_info() if self._cache_enabled else None
        cache_hit = False
        if self._cache_enabled and cache_info_before and cache_info_after:
            cache_hit = cache_info_after.hits > cache_info_before.hits

        result = copy.deepcopy(base_result)
        result.runtime_seconds = perf_counter() - start_time

        if cache_hit:
            message = "Cache hit: returning cached response."
            logger.info("%s query=%r", message, query)
            result.warnings.append(message)
        elif log_experiment:
            self._log_experiment(result)

        return result

    def clear_cache(self) -> None:
        """Clear the cached pipeline responses."""
        if self._cache_enabled:
            self._cached_compute.cache_clear()

    def _compute_result_uncached(self, query: str) -> PipelineResult:
        """Run the full pipeline without caching or experiment logging."""
        start_time = perf_counter()
        result = PipelineResult(query=query, database_path=self.database_path)

        try:
            # Step 1: Time parser
            time_parse_result = self._parse_time_expressions(query, result)
            result.time_parse_result = time_parse_result

            # Step 2: Topic classifier
            classification = self._classify_query(query, result)
            result.query_classification = classification

            # Step 3: Entity linker
            entity_links = self._link_entities(query, result)
            result.entity_links = entity_links

            # Step 4: Text-to-SQL generation with PICARD validation
            generated_sql, validation_status = self._generate_sql(query, result)
            result.generated_sql = generated_sql
            result.validation_status = validation_status

            if not generated_sql:
                result.errors.append("Failed to generate valid SQL")
                result.answer = "I couldn't generate a valid SQL query for your question. Please try rephrasing."
                result.runtime_seconds = perf_counter() - start_time
                return result

            # Step 5: SQL execution
            sql_result = self._execute_sql(generated_sql, result)
            result.sql_execution_result = sql_result

            if sql_result is None or "error" in sql_result:
                result.errors.append(sql_result.get("error", "SQL execution failed"))
                result.answer = "I couldn't execute the query. Please try rephrasing or check the database."
                result.runtime_seconds = perf_counter() - start_time
                return result

            # Step 6: LLM answer generation
            answer = self._generate_answer(query, sql_result, result)
            result.answer = answer

        except Exception as exc:
            logger.exception("Pipeline error for query %r", query)
            result.errors.append(f"Pipeline error: {str(exc)}")
            result.answer = f"I encountered an error processing your query: {str(exc)}"
        finally:
            result.runtime_seconds = perf_counter() - start_time

        return result

    def _parse_time_expressions(self, query: str, result: PipelineResult) -> Optional[Dict[str, Any]]:
        """Extract and parse time expressions from the query."""
        try:
            # Handle explicit year ranges like "between 2015 and 2017" or "from 2015 to 2017"
            range_match = re.search(r"\b(?:between|from)\s+(\d{4})\s+(?:and|to)\s+(\d{4})\b", query, re.IGNORECASE)
            if range_match:
                start_year = int(range_match.group(1))
                end_year = int(range_match.group(2))
                if start_year > end_year:
                    start_year, end_year = end_year, start_year
                expression = range_match.group(0).strip()
                return {
                    "expression": expression,
                    "token": f"range:{start_year}-{end_year}",
                    "granularity": "year",
                    "start_date": f"{start_year}-01-01",
                    "end_date": f"{end_year}-12-31",
                    "all_expressions": [expression],
                }

            # Try to extract time expressions using regex first
            time_patterns = [
                r"\b\d{4}\s*[Qq][1-4]\b",  # 2022 Q3
                r"\b[Qq][1-4]\s*\d{4}\b",  # Q3 2022
                r"\b(last|this|next)\s+(year|quarter|month)\b",  # last quarter
                r"\b\d{4}\b",  # Year
            ]

            found_expressions: list[str] = []
            account_ids_in_query = self._extract_account_ids(query)
            for pattern in time_patterns:
                matches = list(re.finditer(pattern, query, re.IGNORECASE))
                if not matches:
                    continue

                normalized: list[str] = []
                for match in matches:
                    if isinstance(match, re.Match):
                        groups = match.groups()
                        if groups:
                            parts = [part for part in groups if part]
                            if parts:
                                normalized.append(" ".join(parts))
                        else:
                            normalized.append(match.group())
                    else:
                        normalized.append(match)

                for expr in normalized:
                    cleaned = expr.strip()
                    if cleaned and cleaned not in found_expressions:
                        if cleaned.isdigit() and int(cleaned) in account_ids_in_query:
                            # Skip numbers that are actually account ids
                            continue
                        found_expressions.append(cleaned)

            if not found_expressions:
                return None

            # Try to parse the first time expression
            first_expr = found_expressions[0]
            try:
                parse_result = parse_time_expression(first_expr, base_date=self.base_date)
                return {
                    "expression": first_expr,
                    "token": parse_result.to_token(),
                    "granularity": parse_result.granularity,
                    "start_date": parse_result.start_date.isoformat(),
                    "end_date": parse_result.end_date.isoformat(),
                    "all_expressions": found_expressions,
                }
            except TimeParseError:
                return {
                    "expression": first_expr,
                    "parse_error": "Could not parse time expression",
                    "all_expressions": found_expressions,
                }

        except Exception as exc:
            result.warnings.append(f"Time parsing warning: {str(exc)}")
            return None

    def _classify_query(self, query: str, result: PipelineResult) -> Dict[str, Any]:
        """Classify the query topic."""
        try:
            classification = self._query_router.classify(query)
            return {
                "category": classification.category,
                "confidence": classification.confidence,
                "detected_entities": classification.detected_entities,
                "time_expressions": classification.time_expressions,
            }
        except Exception as exc:
            result.warnings.append(f"Query classification warning: {str(exc)}")
            return {"category": "general", "confidence": 0.0, "detected_entities": [], "time_expressions": []}

    def _link_entities(self, query: str, result: PipelineResult) -> Optional[List[Dict[str, Any]]]:
        """Link entities (accounts) using semantic search."""
        try:
            if AccountLinker is None:
                lexical = self._guess_accounts_by_name(query)
                if lexical:
                    result.warnings.append("Entity linking fallback used (lexical search)")
                    return lexical
                result.warnings.append("Entity linking not available (sentence_transformers not installed)")
                return None
            # Extract potential account names from query
            links = self.entity_linker.link_accounts(query, top_k=5, threshold=0.3)
            if links:
                # If exactly one account_id is mentioned explicitly, keep only exact id matches
                explicit_ids = self._extract_account_ids(query)
                if explicit_ids:
                    target_id = explicit_ids[0]
                    filtered = [link for link in links if link.get("account_number") == target_id]
                    if filtered:
                        return filtered
                return links

            lexical = self._guess_accounts_by_name(query)
            if lexical:
                result.warnings.append("Entity linking fallback used (lexical search)")
                return lexical
            return None
        except Exception as exc:
            result.warnings.append(f"Entity linking warning: {str(exc)}")
            return None

    def _generate_sql(self, query: str, result: PipelineResult) -> tuple[Optional[str], Optional[str]]:
        """Generate SQL with PICARD validation."""
        try:
            #Introspect schema from database
            tables = introspect_duckdb_schema(self.database_path)
            if not tables:
                result.errors.append("No tables found in database")
                return None, "no_schema"

            # Generate SQL with validation
            validator = PicardValidator(tables)

            contextual_sql = self._build_contextual_sql(query, result)
            if contextual_sql:
                try:
                    validator.validate(contextual_sql)
                    return contextual_sql, "contextual_primary"
                except PicardValidationError:
                    pass

            generated_sql = self.text_to_sql_generator.generate_sql_with_validation(
                query, tables, validator
            )

            if self._needs_contextual_account_sql(result, generated_sql):
                contextual_sql = self._build_contextual_sql(query, result)
                if contextual_sql:
                    try:
                        validator.validate(contextual_sql)
                        return contextual_sql, "contextual_fallback"
                    except PicardValidationError:
                        pass

            return generated_sql, "validated"
        except PicardValidationError as exc:
            result.errors.append(f"PICARD validation failed: {str(exc)}")
            contextual_sql = self._build_contextual_sql(query, result)
            if contextual_sql:
                try:
                    validator.validate(contextual_sql)
                    return contextual_sql, "contextual_fallback"
                except PicardValidationError:
                    pass
            return None, "validation_failed"
        except Exception as exc:
            result.errors.append(f"SQL generation failed: {str(exc)}")
            return None, "generation_failed"

    def _execute_sql(self, sql: str, result: PipelineResult) -> Optional[Dict[str, Any]]:
        """Execute SQL query on DuckDB."""
        try:
            df = self.sql_executor.run(sql)
            return {
                "rows": len(df),
                "columns": list(df.columns),
                "data": df.to_dict(orient="records"),
            }
        except SQLExecutionTimeout:
            result.errors.append("SQL execution timed out")
            return {"error": "SQL execution timed out"}
        except SQLExecutionError as exc:
            result.errors.append(f"SQL execution error: {str(exc)}")
            return {"error": str(exc)}
        except Exception as exc:
            result.errors.append(f"Unexpected execution error: {str(exc)}")
            return {"error": str(exc)}

    def _generate_answer(
        self, query: str, sql_result: Dict[str, Any], result: PipelineResult
    ) -> str:
        """Generate natural language answer from SQL results."""
        try:
            # Extract metadata for answer generation
            metadata = {}
            if result.time_parse_result:
                metadata["time_expressions"] = result.time_parse_result.get("all_expressions", [])

            # Get dataframe or data
            data = sql_result.get("dataframe") or sql_result.get("data")
            answer = self._answer_generator.generate(query, data, query_metadata=metadata)
            return answer
        except Exception as exc:
            result.warnings.append(f"Answer generation warning: {str(exc)}")
            return f"I found results for your query, but couldn't format them properly: {str(exc)}"

    def _log_experiment(self, result: PipelineResult) -> None:
        """Log experiment for debugging/evaluation."""
        try:
            output_data = result.sql_execution_result.get("data") if result.sql_execution_result else None
            self._experiment_logger.log(
                query=result.query,
                generated_sql=result.generated_sql or "",
                runtime_seconds=result.runtime_seconds,
                output=output_data,
            )
        except Exception:
            # Silently fail logging to avoid breaking the pipeline
            pass

    def _guess_accounts_by_name(self, query: str, limit: int = 5) -> list[Dict[str, Any]]:
        """
        Lightweight lexical search for accounts when embeddings are unavailable.

        Looks for capitalized phrases in the query (e.g., 'Underground Conduit')
        and matches them against account_name with ILIKE.
        """
        try:
            phrases = re.findall(r"([A-Z][a-zA-Z]+(?:\\s+[A-Z][a-zA-Z]+)*)", query)
            phrases = [p.strip() for p in phrases if p.strip()]
            if not phrases:
                return []

            import duckdb  # Local import to avoid hard dependency at module import time

            con = duckdb.connect(self.database_path, read_only=True)
            seen_ids: set[int] = set()
            matches: list[dict[str, Any]] = []

            for phrase in phrases:
                pattern = f"%{phrase.lower()}%"
                rows = con.execute(
                    """
                    SELECT account_id, account_name
                    FROM accounts
                    WHERE lower(account_name) LIKE ?
                    LIMIT ?;
                    """,
                    [pattern, limit],
                ).fetchall()
                for acc_id, acc_name in rows:
                    if acc_id in seen_ids:
                        continue
                    matches.append(
                        {
                            "account_number": acc_id,
                            "confidence": 0.5,
                            "match_type": "lexical",
                            "account_name": acc_name,
                        }
                    )
                    seen_ids.add(acc_id)

            con.close()
            return matches
        except Exception:
            return []

    def _needs_contextual_account_sql(self, result: PipelineResult, sql: Optional[str]) -> bool:
        """Determine if we should replace/generated SQL with an account-aware fallback."""
        explicit_ids = self._extract_account_ids(result.query)
        if not result.entity_links and not explicit_ids:
            return False
        if not sql:
            return True
        if result.entity_links and self._sql_mentions_linked_accounts(sql, result.entity_links):
            return False
        sql_lower = sql.lower()
        return not any(str(acc_id) in sql_lower for acc_id in explicit_ids)

    @staticmethod
    def _sql_mentions_linked_accounts(sql: str, links: Sequence[Dict[str, Any]]) -> bool:
        sql_lower = sql.lower()
        ids = [str(link.get("account_number")) for link in links if link.get("account_number") is not None]
        names = [
            str(link.get("account_name", "")).lower()
            for link in links
            if link.get("account_name")
        ]
        return any(acc_id and acc_id in sql_lower for acc_id in ids) or any(
            name and name in sql_lower for name in names
        )

    @staticmethod
    def _extract_account_ids(query: str) -> list[int]:
        """Pull explicit numeric account references like 'account_id 1840' or 'account 1840'."""
        matches = re.findall(r"\baccount[_\s]?id\s*[:=]?\s*(\d+)", query, re.IGNORECASE)
        matches += re.findall(r"\baccount\s+(\d{3,6})\b", query, re.IGNORECASE)
        ids: list[int] = []
        for m in matches:
            try:
                ids.append(int(m))
            except ValueError:
                continue
        return ids

    @staticmethod
    def _extract_top_k(query: str) -> Optional[int]:
        """Detect top-k intent from the question."""
        lowered = query.lower()
        match = re.search(r"\btop\s+(\d+)", lowered)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        if "top" in lowered or "highest" in lowered or "largest" in lowered:
            return 5
        return None

    def _build_contextual_sql(self, query: str, result: PipelineResult) -> Optional[str]:
        """
        Build a conservative SQL using linked accounts and parsed time ranges.

        This keeps the pipeline accurate when the model omits account filters.
        """
        account_ids: list[int] = []
        account_names: list[str] = []
        top_n = self._extract_top_k(query)
        explicit_ids = self._extract_account_ids(query)
        phrase_ids = self._map_phrases_to_accounts(query)

        if explicit_ids:
            account_ids.extend(explicit_ids)
        elif phrase_ids:
            account_ids.extend(phrase_ids)
        elif top_n is not None:
            # For top-k queries, prefer broad scope unless a specific name is present
            guessed = self._guess_accounts_by_name(query, limit=3)
            for link in guessed:
                acc_id = link.get("account_number")
                acc_name = link.get("account_name")
                if acc_id is not None:
                    account_ids.append(int(acc_id))
                if acc_name:
                    account_names.append(str(acc_name))
        else:
            if result.entity_links:
                links = result.entity_links
                # When no explicit id and a single-account question, prefer the top match only
                selected_links = links[:1]
                for link in selected_links:
                    acc_id = link.get("account_number")
                    acc_name = link.get("account_name")
                    if acc_id is not None:
                        account_ids.append(int(acc_id))
                    if acc_name:
                        account_names.append(str(acc_name))

            if not account_ids and not account_names:
                # Last-resort lexical guess
                guessed = self._guess_accounts_by_name(query, limit=3)
                for link in guessed:
                    acc_id = link.get("account_number")
                    acc_name = link.get("account_name")
                    if acc_id is not None:
                        account_ids.append(int(acc_id))
                    if acc_name:
                        account_names.append(str(acc_name))

        if not account_ids and not account_names and top_n is None:
            return None

        where_clauses: list[str] = []
        if account_ids:
            id_list = ", ".join(str(acc_id) for acc_id in sorted(set(account_ids)))
            where_clauses.append(f"ab.account_id IN ({id_list})")
        if account_names:
            patterns: list[str] = []
            for name in account_names:
                safe = name.lower().replace("'", "''")
                patterns.append(f"LOWER(a.account_name) LIKE '%{safe}%'")
            where_clauses.append("(" + " OR ".join(patterns) + ")")

        if result.time_parse_result:
            start = result.time_parse_result.get("start_date")
            end = result.time_parse_result.get("end_date")
            try:
                if start and end:
                    start_year = int(str(start)[:4])
                    end_year = int(str(end)[:4])
                    if start_year == end_year:
                        where_clauses.append(f"ab.year = {start_year}")
                    else:
                        where_clauses.append(f"ab.year BETWEEN {start_year} AND {end_year}")
            except Exception:
                pass

        where_sql = ""
        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)

        if top_n:
            order_clause = "ORDER BY SUM(ab.amount) DESC, ab.year, a.account_name"
        else:
            order_clause = "ORDER BY ab.year, a.account_name"

        limit_clause = f" LIMIT {top_n}" if top_n else ""

        return (
            "SELECT ab.year, ab.account_id, a.account_name, SUM(ab.amount) AS amount "
            "FROM account_balances AS ab "
            "JOIN accounts AS a ON ab.account_id = a.account_id"
            f"{where_sql} "
            "GROUP BY ab.year, ab.account_id, a.account_name "
            f"{order_clause}{limit_clause};"
        )

    def close(self) -> None:
        """Clean up resources."""
        if self._sql_executor:
            self._sql_executor.close()

    @staticmethod
    def _map_phrases_to_accounts(query: str) -> list[int]:
        """Hard-map specific account phrases to ids to avoid ambiguous linking."""
        q = query.lower()
        mappings = {
            "maintenance of overhead conductors and devices": 5125,
            "overhead conductors and devices": 5125,
            "overhead distribution lines and feeders - right of way": 5135,
            "maintenance of underground conduit": 5145,
            "maintenance of underground conductors and devices": 5150,
            "maintenance of line transformers": 5160,
            "differences between billed and actual settlement amounts for global adjustment": 1589,
            "rsva - global adjustment": 1635,
            "overhead conductors and devices": 1730
        }
        matched: list[int] = []
        for phrase, acc_id in mappings.items():
            if phrase in q:
                matched.append(acc_id)
        return matched


__all__ = ["PipelineOrchestrator", "PipelineResult"]
