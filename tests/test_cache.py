import pytest

from backend.orchestrator import PipelineOrchestrator, PipelineResult
from functools import lru_cache


class DummyOrchestrator(PipelineOrchestrator):
    def __init__(self):
        super().__init__(database_path=":memory:")
        self._compute_calls = 0

        # Replace cached compute with our lightweight implementation
        self._cached_compute = lru_cache(maxsize=8)(self._compute_result_uncached)

    def _compute_result_uncached(self, query: str) -> PipelineResult:
        self._compute_calls += 1
        return PipelineResult(
            query=query,
            database_path=self.database_path,
            answer=f"answer-{self._compute_calls}",
        )


def test_cache_returns_cached_results(monkeypatch):
    orchestrator = DummyOrchestrator()

    first = orchestrator.process_query("What is revenue?", log_experiment=False)
    second = orchestrator.process_query("What is revenue?", log_experiment=False)

    assert first.answer == "answer-1"
    assert second.answer == "answer-1"
    assert orchestrator._compute_calls == 1
    assert any("Cache hit" in warning for warning in second.warnings)


def test_clear_cache_triggers_recomputation():
    orchestrator = DummyOrchestrator()
    orchestrator.process_query("Hello", log_experiment=False)
    orchestrator.clear_cache()
    orchestrator.process_query("Hello", log_experiment=False)
    assert orchestrator._compute_calls == 2





