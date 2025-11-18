import csv
import json

from backend.utils.experiment_logger import ExperimentLogger


def test_experiment_logger_writes_csv(tmp_path):
    log_path = tmp_path / "experiments.csv"
    logger = ExperimentLogger(log_path)

    logger.log(
        query="How many rows?",
        generated_sql="SELECT COUNT(*) FROM items;",
        runtime_seconds=0.123456,
        output=[{"count": 3}],
    )

    with log_path.open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["query"] == "How many rows?"
    assert row["generated_sql"].startswith("SELECT")
    assert float(row["runtime_seconds"]) > 0
    assert json.loads(row["output_json"]) == [{"count": 3}]


def test_experiment_logger_appends_entries(tmp_path):
    log_path = tmp_path / "log.csv"
    logger = ExperimentLogger(log_path)

    logger.log(query="q1", generated_sql="s1", runtime_seconds=0.1, output=None)
    logger.log(query="q2", generated_sql="s2", runtime_seconds=0.2, output={"a": 1})

    with log_path.open() as handle:
        rows = list(csv.DictReader(handle))

    assert [row["query"] for row in rows] == ["q1", "q2"]

