from __future__ import annotations

import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


FIELDNAMES = ["timestamp", "query", "generated_sql", "runtime_seconds", "output_json"]


class ExperimentLogger:
    """
    Lightweight CSV logger for recording text-to-SQL experiments.

    Each entry captures:
        * query            : the natural language question
        * generated_sql    : SQL produced by the model
        * runtime_seconds  : end-to-end runtime
        * output_json      : serialized execution results
    """

    def __init__(self, destination: str | Path = "logs/experiments.csv") -> None:
        self._path = Path(destination)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self._path.exists():
            self._write_header()

    def log(
        self,
        *,
        query: str,
        generated_sql: str,
        runtime_seconds: float,
        output: Any,
    ) -> None:
        payload = {
            "timestamp": self._timestamp(),
            "query": query,
            "generated_sql": generated_sql,
            "runtime_seconds": f"{runtime_seconds:.4f}",
            "output_json": self._serialize_output(output),
        }

        with self._lock, self._path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writerow(payload)

    def _write_header(self) -> None:
        with self._path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _serialize_output(output: Any) -> str:
        if output is None:
            return ""
        if isinstance(output, str):
            return output
        try:
            return json.dumps(output)
        except (TypeError, ValueError):
            return json.dumps(str(output))


__all__ = ["ExperimentLogger"]

