"""
Simple topic classifier/router for financial queries.

Categorizes queries into topic categories to help route processing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

TopicCategory = Literal[
    "financial_query",  # Questions about account balances, revenue, expenses
    "time_based_query",  # Queries with temporal expressions
    "aggregation_query",  # Questions about totals, averages, counts
    "comparison_query",  # Questions comparing periods or accounts
    "general",  # Unclassified queries
]


@dataclass(frozen=True)
class QueryClassification:
    category: TopicCategory
    confidence: float
    detected_entities: list[str]
    time_expressions: list[str]


class QueryRouter:
    """
    Simple rule-based topic classifier for financial queries.

    Uses keyword matching and pattern detection to categorize queries.
    """

    FINANCIAL_KEYWORDS = {
        "revenue", "income", "expense", "profit", "loss", "balance", "account",
        "sales", "revenue", "cost", "asset", "liability", "equity", "debit", "credit",
        "amount", "total", "sum", "payment", "transaction", "financial",
    }

    TIME_KEYWORDS = {
        "year", "quarter", "month", "day", "period", "fiscal", "annual",
        "monthly", "weekly", "daily", "last", "this", "next", "previous",
    }

    AGGREGATION_KEYWORDS = {
        "total", "sum", "average", "avg", "mean", "count", "maximum", "minimum",
        "max", "min", "aggregate", "group", "per", "each",
    }

    COMPARISON_KEYWORDS = {
        "compare", "comparison", "vs", "versus", "against", "difference",
        "compared to", "than", "more than", "less than", "higher", "lower",
    }

    def classify(self, query: str) -> QueryClassification:
        """
        Classify a query into topic categories.

        Args:
            query: Natural language query string

        Returns:
            QueryClassification with category, confidence, and detected entities
        """
        query_lower = query.lower()
        words = set(re.findall(r"\b\w+\b", query_lower))

        # Detect entities
        detected_entities = []
        time_expressions = []

        # Time expression detection
        time_patterns = [
            r"\b\d{4}\s*[Qq][1-4]\b",  # 2022 Q3
            r"\b[Qq][1-4]\s*\d{4}\b",  # Q3 2022
            r"\b(last|this|next)\s+(year|quarter|month)\b",  # last quarter
            r"\b\d{4}\b",  # Year
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b",  # March 2023
        ]
        for pattern in time_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                time_expressions.extend(matches if isinstance(matches[0], str) else ["".join(m) for m in matches])

        # Financial entity detection
        financial_matches = words.intersection(self.FINANCIAL_KEYWORDS)
        detected_entities.extend(financial_matches)

        # Classification logic
        category_scores = {
            "financial_query": len(words.intersection(self.FINANCIAL_KEYWORDS)),
            "time_based_query": len(time_expressions) + len(words.intersection(self.TIME_KEYWORDS)),
            "aggregation_query": len(words.intersection(self.AGGREGATION_KEYWORDS)),
            "comparison_query": len(words.intersection(self.COMPARISON_KEYWORDS)),
        }

        # Determine primary category
        max_score = max(category_scores.values()) if category_scores.values() else 0
        primary_category = "financial_query" if max_score > 0 else "general"

        for cat, score in category_scores.items():
            if score > 0 and score >= max_score:
                primary_category = cat
                break

        # Calculate confidence based on keyword matches
        total_matches = sum(category_scores.values())
        confidence = min(1.0, total_matches / max(3, len(words))) if words else 0.3

        return QueryClassification(
            category=primary_category,
            confidence=confidence,
            detected_entities=list(detected_entities),
            time_expressions=list(set(time_expressions)),
        )


__all__ = ["QueryRouter", "QueryClassification", "TopicCategory"]
