"""
Hybrid query router for deciding between Text-to-SQL, RAG, or refusal.

This component combines:
- Lightweight regex/keyword heuristics for numeric and financial cues
- Embedding similarity against two label prototypes (text_to_sql vs rag)

It returns a structured decision with:
- route: 'text_to_sql', 'rag', or 'refusal'
- confidence: overall confidence score in [0, 1]
- numeric_cues: extracted numeric / account signals
- similarities: embedding similarities per label
- notes: human-readable explanation for logging / debugging
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Literal, Optional
import logging
import math
import re

try:
    # Reuse sentence-transformers if available in the environment
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


logger = logging.getLogger(__name__)


RouteLabel = Literal["text_to_sql", "rag", "refusal"]


@dataclass(frozen=True)
class HybridRouteDecision:
    """Structured decision from the hybrid router."""

    route: RouteLabel
    confidence: float
    numeric_cues: Dict[str, float]
    similarities: Dict[str, float]
    notes: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class HybridRouter:
    """
    Hybrid router that combines heuristics and embeddings.

    High-level logic:
    - Extract numeric / financial cues (accounts, years, amounts, top-k).
    - Compute heuristic scores for 'text_to_sql' vs 'rag'.
    - Optionally, use embeddings to compare query against two prototypes.
    - Combine scores into a final decision and confidence.
    - Fall back gracefully when embeddings are unavailable.
    """

    # Prototypes for embedding similarity (can be tuned later)
    _SQL_PROTO = (
        "Ask for numeric financial values using accounts, years, balances, amounts, "
        "and filters that should be answered from a structured SQL database."
    )
    _RAG_PROTO = (
        "Ask conceptual financial questions, definitions, explanations, accounting "
        "policies, or qualitative comparisons that require textual knowledge."
    )

    # Simple keyword sets
    _FINANCE_KEYWORDS = {
        "revenue",
        "income",
        "expense",
        "profit",
        "loss",
        "balance",
        "account",
        "cash",
        "asset",
        "liability",
        "equity",
        "debit",
        "credit",
        "trial",
        "ledger",
        "amount",
        "total",
        "sum",
    }

    _CONCEPT_KEYWORDS = {
        "what is",
        "what are",
        "explain",
        "definition",
        "define",
        "how does",
        "why does",
        "difference between",
        "compare",
    }

    # Thresholds (can be tuned)
    _LOW_CONF_THRESHOLD = 0.35

    def __init__(self, *, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._embed_model: Optional[SentenceTransformer] = None

    # -------------------- public API --------------------

    def route(self, query: str) -> HybridRouteDecision:
        """
        Decide which path to use for a given query.

        Returns:
            HybridRouteDecision with route in {'text_to_sql', 'rag', 'refusal'}.
        """
        query = (query or "").strip()
        if not query:
            decision = HybridRouteDecision(
                route="refusal",
                confidence=0.0,
                numeric_cues={},
                similarities={},
                notes="Empty query.",
            )
            logger.info("HybridRouter decision: %s", decision.to_dict())
            return decision

        numeric_cues = self._extract_numeric_cues(query)
        heuristic_sql_score, heuristic_rag_score = self._compute_heuristic_scores(
            query, numeric_cues
        )

        sim_sql, sim_rag = self._compute_label_similarities(query)

        similarities = {
            "text_to_sql": sim_sql,
            "rag": sim_rag,
        }

        score_sql, score_rag = self._combine_scores(
            heuristic_sql_score, heuristic_rag_score, sim_sql, sim_rag
        )

        best_label: RouteLabel
        best_score = max(score_sql, score_rag)
        if best_score < self._LOW_CONF_THRESHOLD:
            best_label = "refusal"
        else:
            best_label = "text_to_sql" if score_sql >= score_rag else "rag"

        notes_parts = [
            f"heuristic_sql={heuristic_sql_score:.3f}",
            f"heuristic_rag={heuristic_rag_score:.3f}",
            f"sim_sql={sim_sql:.3f}",
            f"sim_rag={sim_rag:.3f}",
        ]
        notes = "; ".join(notes_parts)

        decision = HybridRouteDecision(
            route=best_label,
            confidence=float(max(0.0, min(1.0, best_score))),
            numeric_cues=numeric_cues,
            similarities=similarities,
            notes=notes,
        )
        logger.info("HybridRouter decision: %s", decision.to_dict())
        return decision

    # -------------------- internals --------------------

    @staticmethod
    def _extract_numeric_cues(query: str) -> Dict[str, float]:
        """Extract counts of numeric cues and financial patterns."""
        lowered = query.lower()

        # Years like 2019, 2020, etc.
        years = re.findall(r"\b(19|20)\d{2}\b", query)

        # Account-like IDs
        account_id_matches = re.findall(
            r"\baccount[_\s]?id\s*[:=]?\s*(\d+)", query, flags=re.IGNORECASE
        )
        account_short_matches = re.findall(
            r"\baccount\s+(\d{3,6})\b", query, flags=re.IGNORECASE
        )

        # Generic numeric tokens
        numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", query)

        # Top-k style
        top_k_match = re.search(r"\btop\s+(\d+)\b", lowered)

        # Finance keywords
        finance_hits = sum(1 for kw in HybridRouter._FINANCE_KEYWORDS if kw in lowered)

        return {
            "num_years": float(len(years)),
            "num_account_ids": float(len(account_id_matches) + len(account_short_matches)),
            "num_numbers": float(len(numbers)),
            "has_top_k": 1.0 if top_k_match else 0.0,
            "finance_keywords": float(finance_hits),
        }

    def _compute_heuristic_scores(
        self, query: str, cues: Dict[str, float]
    ) -> tuple[float, float]:
        """Compute heuristic scores for text_to_sql vs rag based on cues and phrases."""
        lowered = query.lower()

        # SQL-like heuristic: many numbers, years, accounts, finance keywords
        sql_score = (
            0.25 * cues.get("num_numbers", 0.0)
            + 0.5 * cues.get("num_years", 0.0)
            + 0.7 * cues.get("num_account_ids", 0.0)
            + 0.2 * cues.get("has_top_k", 0.0)
            + 0.1 * cues.get("finance_keywords", 0.0)
        )

        # Conceptual / RAG heuristic: "what is", "explain", etc., and few numbers
        concept_hits = sum(
            1.0 for phrase in self._CONCEPT_KEYWORDS if phrase in lowered
        )
        # Penalize heavy numeric presence for RAG
        numeric_penalty = 0.2 * cues.get("num_numbers", 0.0)
        rag_score = max(0.0, concept_hits - numeric_penalty)

        # Squash to [0, 1]-ish using 1 - exp(-x)
        sql_score_norm = 1.0 - math.exp(-sql_score)
        rag_score_norm = 1.0 - math.exp(-rag_score)
        return sql_score_norm, rag_score_norm

    def _compute_label_similarities(self, query: str) -> tuple[float, float]:
        """
        Compute embedding-based similarities between the query and label prototypes.

        Returns:
            (sim_text_to_sql, sim_rag) in [0, 1]. When embeddings are unavailable,
            returns (0.0, 0.0) so routing falls back to heuristics only.
        """
        if SentenceTransformer is None:
            return 0.0, 0.0

        try:
            if self._embed_model is None:
                self._embed_model = SentenceTransformer(self._model_name)

            texts = [query, self._SQL_PROTO, self._RAG_PROTO]
            embeddings = self._embed_model.encode(texts, normalize_embeddings=True)
            q_vec, sql_vec, rag_vec = embeddings

            # Cosine with normalized vectors is just dot product in [-1, 1]; clamp to [0, 1]
            sim_sql = float(max(0.0, min(1.0, float(q_vec @ sql_vec))))
            sim_rag = float(max(0.0, min(1.0, float(q_vec @ rag_vec))))
            return sim_sql, sim_rag
        except Exception as exc:  # pragma: no cover - safety net
            logger.warning("HybridRouter embedding similarity failed: %s", exc)
            return 0.0, 0.0

    @staticmethod
    def _combine_scores(
        heuristic_sql: float,
        heuristic_rag: float,
        sim_sql: float,
        sim_rag: float,
    ) -> tuple[float, float]:
        """
        Combine heuristic and embedding scores into final scores.

        Simple linear blend with slightly higher weight on heuristics so that
        routing still works when embeddings are noisy.
        """
        w_heur = 0.65
        w_embed = 0.35

        score_sql = w_heur * heuristic_sql + w_embed * sim_sql
        score_rag = w_heur * heuristic_rag + w_embed * sim_rag
        return score_sql, score_rag


def build_refusal_message(query: str, decision: HybridRouteDecision) -> str:
    """
    Build a helpful refusal message with guidance based on the router decision.
    
    Args:
        query: The original user query
        decision: The router decision that led to refusal
        
    Returns:
        A user-friendly refusal message with guidance on how to rephrase
    """
    confidence = decision.confidence
    numeric_cues = decision.numeric_cues
    similarities = decision.similarities
    
    # Check if query is completely off-topic (very low similarity to both prototypes)
    max_similarity = max(similarities.get("text_to_sql", 0.0), similarities.get("rag", 0.0))
    
    # Check if query has any financial/numeric signals
    has_account_id = numeric_cues.get("account_id_count", 0) > 0
    has_year = numeric_cues.get("year_count", 0) > 0
    has_financial_keywords = numeric_cues.get("financial_keyword_score", 0.0) > 0.1
    
    # Build contextual guidance based on what's missing
    guidance_parts = []
    
    if max_similarity < 0.3:
        # Completely off-topic query
        return (
            f"I'm not able to answer '{query}' because it doesn't appear to be related "
            "to the financial data available in this system. I can help with questions about "
            "financial account balances and transactions, revenue/expenses/profit analysis, "
            "account-specific queries with account IDs or names, and time-based financial comparisons "
            "(e.g., 'revenue in 2023'). Please rephrase your question to focus on financial data queries."
        )
    
    if confidence < 0.4:
        # Low confidence - query is ambiguous or unclear
        if not has_account_id and not has_year and not has_financial_keywords:
            return (
                f"I'm not confident I can answer '{query}' accurately. To help me better understand "
                "your question, please include specific account IDs or account names (e.g., 'account 1840'), "
                "time periods (e.g., 'in 2023', 'between 2020 and 2022'), and financial metrics "
                "(e.g., 'revenue', 'expenses', 'balance'). Example: 'What was the total revenue for account 1840 in 2023?'"
            )
        else:
            # Has some signals but still low confidence
            missing = []
            if not has_account_id:
                missing.append("specific account IDs or names")
            if not has_year:
                missing.append("time periods (years, quarters)")
            if not has_financial_keywords:
                missing.append("financial terms (revenue, expenses, balance)")
            
            if missing:
                guidance = ", ".join(missing[:-1])
                if len(missing) > 1:
                    guidance += f", or {missing[-1]}"
                else:
                    guidance = missing[0]
                
                return (
                    f"I'm not confident I can answer '{query}' with the available information. "
                    f"Please try including {guidance} in your question. "
                    "Example: 'What was the total revenue for account 1840 in 2023?'"
                )
    
    # Generic low confidence refusal
    return (
        f"I'm not able to answer '{query}' reliably with the available financial data. "
        "Please try rephrasing your question with specific account IDs or account names, "
        "time periods (years, quarters), and clear financial metrics (revenue, expenses, balance, profit). "
        "Example: 'What was the total revenue for account 1840 in 2023?'"
    )


__all__ = ["HybridRouter", "HybridRouteDecision", "RouteLabel", "build_refusal_message"]


