import os

import duckdb
import pandas as pd
import pytest

from backend.text_to_sql import (
    PicardValidator,
    PicardValidationError,
    TableSchema,
    TextToSQLGenerator,
    run_toy_example,
)
from backend.utils.experiment_logger import ExperimentLogger


@pytest.fixture(scope="module")
def toy_schema():
    return [TableSchema(name="sales", columns=["product", "region", "amount"])]


def test_picard_validator_rejects_unknown_table(toy_schema):
    validator = PicardValidator(toy_schema)
    with pytest.raises(PicardValidationError):
        validator.validate("SELECT * FROM unknown;")


def test_text_to_sql_generation_and_validation(toy_schema, tmp_path):
    generator = TextToSQLGenerator()
    validator = PicardValidator(toy_schema)
    sql_query = generator.generate_sql_with_validation(
        "List total amount for each product",
        toy_schema,
        validator,
    )

    con = duckdb.connect()
    con.execute("CREATE TABLE sales (product VARCHAR, region VARCHAR, amount INTEGER);")
    con.execute("INSERT INTO sales VALUES ('Laptop', 'NA', 1200), ('Laptop', 'EU', 1100);")

    df = con.execute(sql_query).df()
    con.close()

    assert {"product", "total_amount"} <= set(map(str.lower, df.columns))


def test_run_toy_example_creates_results(tmp_path):
    logger = ExperimentLogger(tmp_path / "runs.csv")
    artifacts = run_toy_example(logger=logger)
    assert artifacts["validated_sql"]
    assert os.path.exists(artifacts["results_path"])
    df = pd.read_csv(artifacts["results_path"])
    assert not df.empty
    with (tmp_path / "runs.csv").open() as handle:
        assert len(handle.readlines()) >= 2

