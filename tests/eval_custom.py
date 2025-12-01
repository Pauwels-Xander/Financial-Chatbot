import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb
from backend.orchestrator import PipelineOrchestrator


def _extract_numeric(text: str) -> Optional[float]:
    """
    Extract the most likely metric value from an answer string.
    Skips year-like tokens and handles K/M suffixes.
    """
    if not text:
        return None
    candidates: list[float] = []
    for match in re.finditer(r"\$?([-+]?\d*[.,]?\d+(?:[eE][-+]?\d+)?)([KkMm]?)", str(text)):
        raw = match.group(1).replace(",", "")
        suffix = match.group(2).lower()
        try:
            val = float(raw)
        except ValueError:
            continue
        # ignore likely years
        if 1900 <= val <= 2100:
            continue
        if suffix == "k":
            val *= 1_000
        elif suffix == "m":
            val *= 1_000_000
        candidates.append(val)
    if not candidates:
        return None
    return max(candidates, key=lambda x: abs(x))


def evaluate(dataset_path: Path, output_path: Path, *, run_oracle_sql: bool = False) -> List[Dict]:
    with dataset_path.open("r", encoding="utf-8") as f:
        qa_set: List[Dict] = json.load(f)

    pipeline = PipelineOrchestrator(database_path="data/db/trial_balance.duckdb")
    results: List[Dict] = []

    for item in qa_set:
        question = item["question"]
        true_value = float(item["answer"])

        start = time.time()
        output = pipeline.process_query(question, log_experiment=False)
        latency = time.time() - start

        predicted_value = _extract_numeric(output.answer)

        absolute_match = None
        relative_error = None
        if predicted_value is not None:
            absolute_match = abs(predicted_value - true_value) < 1e-6
            if true_value != 0:
                relative_error = abs(predicted_value - true_value) / abs(true_value)

        oracle_value = None
        if run_oracle_sql:
            try:
                con = duckdb.connect("data/db/trial_balance.duckdb", read_only=True)
                df = con.execute(item["sql"]).df()
                con.close()
                if not df.empty and df.shape[1] > 0:
                    oracle_value = float(df.iloc[0, 0])
            except Exception:
                oracle_value = None

        results.append(
            {
                "question": question,
                "answer": output.answer,
                "predicted_value": predicted_value,
                "true_value": true_value,
                "oracle_value": oracle_value,
                "absolute_match": absolute_match,
                "relative_error": relative_error,
                "generated_sql": output.generated_sql,
                "validation_status": output.validation_status,
                "latency_sec": latency,
            }
        )

    pipeline.close()

    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "question",
            "answer",
            "predicted_value",
            "true_value",
            "oracle_value",
            "absolute_match",
            "relative_error",
            "generated_sql",
            "validation_status",
            "latency_sec",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    return results


def summarize(results: List[Dict]) -> Dict[str, float]:
    total = len(results)
    abs_matches = sum(1 for r in results if r.get("absolute_match"))
    rel_errors = [r["relative_error"] for r in results if r.get("relative_error") is not None]
    avg_rel_error = sum(rel_errors) / len(rel_errors) if rel_errors else None
    return {
        "total": total,
        "absolute_matches": abs_matches,
        "avg_relative_error": avg_rel_error,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pipeline against custom QA set.")
    parser.add_argument("--dataset", type=Path, default=Path("tests/qa_set_custom.json"))
    parser.add_argument("--output", type=Path, default=Path("tests/evaluation_results_custom.csv"))
    parser.add_argument("--oracle", action="store_true", help="Also run ground-truth SQL via DuckDB for comparison.")
    args = parser.parse_args()

    results = evaluate(args.dataset, args.output, run_oracle_sql=args.oracle)
    summary = summarize(results)
    print("Wrote detailed results to", args.output)
    print("Summary:", summary)
