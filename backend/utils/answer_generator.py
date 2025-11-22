"""
LLM-based answer generator for converting SQL results into natural language responses.

This module provides a simple template-based answer generator that can be replaced
with an actual LLM (GPT-4, LLaMA, etc.) for more sophisticated responses.
"""

from __future__ import annotations

import json
from numbers import Number
from typing import Any, Dict, List, Optional

import pandas as pd


class AnswerGenerator:
    """
    Generates natural language answers from SQL query results.

    For now, uses template-based formatting. Can be extended to use
    an actual LLM for more sophisticated responses.
    """

    def generate(
        self,
        query: str,
        sql_result: pd.DataFrame | Dict[str, Any] | List[Dict[str, Any]] | str,
        *,
        query_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a natural language answer from SQL results.

        Args:
            query: Original natural language query
            sql_result: Query results (DataFrame, dict, list, or JSON string)
            query_metadata: Optional metadata about the query (time expressions, etc.)

        Returns:
            Natural language answer string
        """
        # Normalize input
        if isinstance(sql_result, str):
            try:
                sql_result = json.loads(sql_result)
            except json.JSONDecodeError:
                return f"I found results for your query '{query}', but couldn't format them properly."

        if isinstance(sql_result, pd.DataFrame):
            if sql_result.empty:
                return f"I couldn't find any results for '{query}'."
            sql_result = sql_result.to_dict(orient="records")

        if isinstance(sql_result, dict):
            # Single row result
            return self._format_single_result(query, sql_result, query_metadata)

        if isinstance(sql_result, list):
            if not sql_result:
                return f"I couldn't find any results for '{query}'."
            if len(sql_result) == 1:
                return self._format_single_result(query, sql_result[0], query_metadata)
            return self._format_multiple_results(query, sql_result, query_metadata)

        return f"I found results for '{query}', but couldn't format them properly."

    def build_numeric_llm_prompt(
        self,
        query: str,
        sql_result: pd.DataFrame | Dict[str, Any] | List[Dict[str, Any]] | str,
        *,
        query_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build a reusable LLM prompt that formats numeric SQL results.

        The prompt asks the LLM to return a compact JSON payload containing:
        - value: formatted headline number (latest period when present)
        - trend: up/down/flat/n/a based on the last two periods
        - percent_change: signed percent change when two periods exist, else "n/a"
        - summary: short natural-language sentence describing the number and trend
        """
        records = self._normalize_records(sql_result)
        metadata = query_metadata or {}
        metric_key = self._select_numeric_field(records)
        period_key = self._select_period_field(records)
        trend_snapshot = self._compute_trend_snapshot(records, metric_key, period_key)

        serialized_rows = json.dumps(records, indent=2, default=str)
        serialized_context = json.dumps(
            {
                "question": query,
                "metric_key": metric_key,
                "period_column": period_key,
                "trend_snapshot": trend_snapshot,
                "metadata": metadata,
            },
            indent=2,
            default=str,
        )

        return (
            "You are a financial analysis formatter. "
            "Given SQL result rows, return concise JSON with a headline value, trend, percent change, and a one-sentence summary.\n"
            f"Question:\n{query}\n\n"
            f"SQL result rows (JSON):\n{serialized_rows}\n\n"
            f"Derived context:\n{serialized_context}\n\n"
            "Instructions:\n"
            "- Always reply with JSON only. Keys: value, trend, percent_change, summary.\n"
            "- Use the latest period as the headline when a period column is available; otherwise use the first numeric metric.\n"
            "- Format value with a dollar sign and commas when the metric suggests money (amount, total, revenue, expense, profit, balance); "
            "otherwise use commas for thousands. Keep at most two decimal places.\n"
            "- Trend should be one of up, down, flat, or n/a. When two periods exist, compare the latest value to the previous one.\n"
            "- Percent change should use the last two periods when available; sign it (e.g., +4.2% or -3.1%). Use 'n/a' when a change cannot be computed.\n"
            "- Summary should be a short sentence that mentions the value and, when present, the period comparison.\n"
            "Example output:\n"
            '{\"value\": \"$1.20M\", \"trend\": \"up\", \"percent_change\": \"+8.5%\", '
            '\"summary\": \"Revenue rose 8.5% to $1.20M in 2023 versus 2022.\"}'
        )

    def _format_single_result(self, query: str, result: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> str:
        """Format a single-row result into a natural language answer."""
        # Extract numeric values
        numeric_values = {}
        text_values = {}
        for key, value in result.items():
            if isinstance(value, (int, float)):
                numeric_values[key] = value
            else:
                text_values[key] = str(value) if value is not None else "N/A"

        # Build answer based on query type
        query_lower = query.lower()
        
        # Time-based queries
        if metadata and metadata.get("time_expressions"):
            time_info = ", ".join(metadata["time_expressions"])
            if numeric_values:
                value_key = list(numeric_values.keys())[0]
                value = numeric_values[value_key]
                formatted_value = self._format_number(value)
                return f"For {time_info}, {value_key.replace('_', ' ')} was {formatted_value}."

        # Total/Sum queries
        if "total" in query_lower or "sum" in query_lower:
            if numeric_values:
                value_key = list(numeric_values.keys())[0]
                value = numeric_values[value_key]
                formatted_value = self._format_number(value)
                return f"The total {value_key.replace('_', ' ')} is {formatted_value}."

        # Count queries
        if "how many" in query_lower or "count" in query_lower:
            if numeric_values:
                count = list(numeric_values.values())[0]
                return f"I found {count} result(s)."

        # Default: format as key-value pairs
        parts = []
        for key, value in {**numeric_values, **text_values}.items():
            if isinstance(value, (int, float)):
                parts.append(f"{key.replace('_', ' ')}: {self._format_number(value)}")
            else:
                parts.append(f"{key.replace('_', ' ')}: {value}")

        return f"For your query '{query}', I found: {', '.join(parts)}."

    def _format_multiple_results(self, query: str, results: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]]) -> str:
        """Format multiple-row results into a natural language answer."""
        if not results:
            return f"I couldn't find any results for '{query}'."

        # Try to identify a summary column (total, sum, amount, etc.)
        summary_columns = ["total", "sum", "amount", "count", "value"]
        summary_key = None
        for col in summary_columns:
            if col in results[0]:
                summary_key = col
                break

        if summary_key:
            total = sum(
                result[summary_key] for result in results
                if isinstance(result.get(summary_key), (int, float))
            )
            formatted_total = self._format_number(total)
            return f"I found {len(results)} results. The total {summary_key.replace('_', ' ')} is {formatted_total}."

        # Group by common dimensions
        if len(results) <= 5:
            parts = []
            for i, result in enumerate(results, 1):
                parts.append(f"Result {i}: {', '.join(f'{k}={v}' for k, v in result.items())}")
            return f"I found {len(results)} results:\n" + "\n".join(parts)

        return f"I found {len(results)} results for '{query}'."

    @staticmethod
    def _format_number(value: float | int) -> str:
        """Format a number with appropriate precision and units."""
        if abs(value) >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"
        if abs(value) >= 1_000:
            return f"${value / 1_000:.2f}K"
        if isinstance(value, float):
            return f"${value:,.2f}"
        return f"${value:,}"

    def _normalize_records(
        self,
        sql_result: pd.DataFrame | Dict[str, Any] | List[Dict[str, Any]] | str,
    ) -> List[Dict[str, Any]]:
        """Normalize different result types into a list of dict records for prompting."""
        data: List[Dict[str, Any]]
        if isinstance(sql_result, str):
            try:
                parsed = json.loads(sql_result)
            except json.JSONDecodeError as exc:
                raise ValueError("SQL result string must be valid JSON for prompt building") from exc
            if isinstance(parsed, dict):
                data = [parsed]
            elif isinstance(parsed, list):
                data = parsed
            else:
                raise ValueError("SQL result JSON must decode to an object or array")
        elif isinstance(sql_result, pd.DataFrame):
            data = sql_result.to_dict(orient="records")
        elif isinstance(sql_result, dict):
            data = [sql_result]
        elif isinstance(sql_result, list):
            data = sql_result
        else:
            raise ValueError("Unsupported SQL result type for prompt building")

        normalized: List[Dict[str, Any]] = []
        for row in data:
            if not isinstance(row, dict):
                raise ValueError("SQL result rows must be dictionaries for prompt building")
            normalized.append(dict(row))
        return normalized

    @staticmethod
    def _select_numeric_field(rows: List[Dict[str, Any]]) -> Optional[str]:
        """Pick a primary numeric field to headline in the prompt."""
        if not rows:
            return None

        priority_hints = [
            "amount",
            "total",
            "value",
            "revenue",
            "expense",
            "profit",
            "balance",
            "count",
        ]

        numeric_candidates: List[str] = []
        first_row = rows[0]
        for key in first_row.keys():
            values = [row.get(key) for row in rows]
            if all(AnswerGenerator._coerce_number(v) is not None for v in values if v is not None):
                numeric_candidates.append(key)

        for hint in priority_hints:
            for key in numeric_candidates:
                if hint in key.lower():
                    return key

        return numeric_candidates[0] if numeric_candidates else None

    @staticmethod
    def _select_period_field(rows: List[Dict[str, Any]]) -> Optional[str]:
        """Identify a period-like field (year, quarter, month, date) if present."""
        if not rows:
            return None
        period_hints = ["period", "date", "year", "month", "quarter", "fiscal"]
        for key in rows[0].keys():
            lowered = key.lower()
            if any(hint in lowered for hint in period_hints):
                return key
        return None

    @staticmethod
    def _coerce_number(value: Any) -> Optional[float]:
        """Convert a value to float when possible."""
        if value is None:
            return None
        if isinstance(value, Number):
            return float(value)
        if isinstance(value, str):
            try:
                sanitized = value.replace(",", "")
                return float(sanitized)
            except ValueError:
                return None
        return None

    def _compute_trend_snapshot(
        self,
        rows: List[Dict[str, Any]],
        metric_key: Optional[str],
        period_key: Optional[str],
    ) -> Dict[str, Any]:
        """
        Compute the latest/previous values, trend direction, and percent change.

        Returns a dictionary suitable for embedding in the LLM prompt so the
        model can mirror these pre-computed numbers.
        """
        snapshot: Dict[str, Any] = {
            "metric_key": metric_key,
            "period_column": period_key,
            "latest_value": None,
            "latest_period": None,
            "previous_value": None,
            "previous_period": None,
            "trend": "n/a",
            "percent_change": None,
        }

        if not metric_key or not rows:
            return snapshot

        ordered_rows = rows
        if period_key:
            ordered_rows = sorted(rows, key=lambda row: str(row.get(period_key)))

        latest_row = ordered_rows[-1]
        prev_row = ordered_rows[-2] if len(ordered_rows) >= 2 else None

        latest_value = self._coerce_number(latest_row.get(metric_key))
        previous_value = self._coerce_number(prev_row.get(metric_key)) if prev_row else None

        snapshot["latest_value"] = latest_value
        snapshot["latest_period"] = latest_row.get(period_key) if period_key else None
        snapshot["previous_value"] = previous_value
        snapshot["previous_period"] = prev_row.get(period_key) if (prev_row and period_key) else None

        if latest_value is None or previous_value is None:
            return snapshot

        if latest_value > previous_value:
            snapshot["trend"] = "up"
        elif latest_value < previous_value:
            snapshot["trend"] = "down"
        else:
            snapshot["trend"] = "flat"

        if previous_value != 0:
            percent_change = round(((latest_value - previous_value) / abs(previous_value)) * 100, 2)
            snapshot["percent_change"] = percent_change

        return snapshot


__all__ = ["AnswerGenerator"]

