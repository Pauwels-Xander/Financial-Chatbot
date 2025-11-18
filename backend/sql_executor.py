from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, Dict, Optional

import duckdb
import pandas as pd


class SQLExecutionError(Exception):
    """Raised when DuckDB cannot complete a query."""


class SQLExecutionTimeout(SQLExecutionError):
    """Raised when a query exceeds the allowed runtime."""


@dataclass(frozen=True)
class QueryResult:
    dataframe: pd.DataFrame

    def to_json(self, *, orient: str = "records", date_format: str = "iso") -> str:
        return self.dataframe.to_json(orient=orient, date_format=date_format)


class DuckDBExecutor:
    """
    Safe, short-lived wrapper around DuckDB query execution.

    Each query runs in its own connection, with a timeout enforced via a worker thread.
    Exceptions from DuckDB are caught and normalized into SQLExecutionError subclasses.
    """

    def __init__(
        self,
        database_path: str,
        *,
        default_timeout: float = 5.0,
        max_workers: int = 1,
        row_limit: Optional[int] = None,
    ) -> None:
        """
        Args:
            database_path: Path to the DuckDB database file.
            default_timeout: Seconds to wait before aborting a query when no explicit timeout is provided.
            max_workers: Number of workers dedicated to query execution.
            row_limit: Optional ceiling on rows returned per query.
        """
        if default_timeout <= 0:
            raise ValueError("default_timeout must be greater than 0")

        self._database_path = database_path
        self._default_timeout = default_timeout
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="duckdb-query")
        self._row_limit = row_limit

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def run(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
        as_json: bool = False,
        orient: str = "records",
        date_format: str = "iso",
    ) -> pd.DataFrame | str:
        """
        Execute a SQL statement and return either a DataFrame or JSON string.

        Args:
            query: SQL text, typically a SELECT.
            params: Optional dictionary of named parameters for DuckDB execute.
            timeout: Seconds before timing out (defaults to self._default_timeout).
            as_json: Return a JSON string rather than a DataFrame when True.
            orient: JSON orientation passed to pandas.DataFrame.to_json.
            date_format: Date formatting for JSON output.

        Raises:
            SQLExecutionTimeout: When execution exceeds the timeout.
            SQLExecutionError: If DuckDB raises any error.
        """
        timeout_value = timeout or self._default_timeout

        try:
            future = self._executor.submit(self._execute_query, query, params)
            result = future.result(timeout=timeout_value)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise SQLExecutionTimeout("Query exceeded the allowed runtime") from exc
        except duckdb.Error as exc:
            raise SQLExecutionError(str(exc)) from exc

        dataframe = result.dataframe
        if as_json:
            return dataframe.to_json(orient=orient, date_format=date_format)

        return dataframe

    def _execute_query(self, query: str, params: Optional[Dict[str, Any]]) -> QueryResult:
        connection = duckdb.connect(self._database_path, read_only=False)

        try:
            if params is None:
                cursor = connection.execute(query)
            else:
                cursor = connection.execute(query, params)
            if self._row_limit is not None and self._row_limit > 0:
                dataframe = cursor.df().head(self._row_limit)
            else:
                dataframe = cursor.df()
            return QueryResult(dataframe)
        finally:
            connection.close()
