"""
LLM-based answer generator for converting SQL results into natural language responses.

This module provides a simple template-based answer generator that can be replaced
with an actual LLM (GPT-4, LLaMA, etc.) for more sophisticated responses.
"""

from __future__ import annotations

import json
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


__all__ = ["AnswerGenerator"]

