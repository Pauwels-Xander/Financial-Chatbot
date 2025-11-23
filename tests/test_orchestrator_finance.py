import duckdb

from backend.orchestrator import PipelineOrchestrator
from backend.text_to_sql import PicardValidator


def _create_finance_db(path: str) -> None:
    con = duckdb.connect(path)
    con.execute(
        """
        CREATE TABLE financials (
            Current_Company_Name VARCHAR,
            Year INTEGER,
            Cash_and_Cash_Equivalents_USofA_1005_1070 DECIMAL(18, 2),
            Net_Income_Total DECIMAL(18, 2)
        );
        """
    )
    con.execute(
        """
        INSERT INTO financials VALUES
            ('Utility A', 2022, 1200.50, 300.25),
            ('Utility A', 2023, 1400.75, 350.75);
        """
    )
    con.close()


class StubTextToSQLGenerator:
    def __init__(self, sql: str):
        self.sql = sql

    def generate_sql_with_validation(self, query, tables, validator: PicardValidator):
        validator.validate(self.sql)
        return self.sql


def test_orchestrator_runs_on_financials(tmp_path):
    db_path = tmp_path / "finance.duckdb"
    _create_finance_db(str(db_path))

    stub_sql = "SELECT Year, Net_Income_Total FROM financials ORDER BY Year;"
    orchestrator = PipelineOrchestrator(
        str(db_path),
        text_to_sql_generator=StubTextToSQLGenerator(stub_sql),
    )

    result = orchestrator.process_query("Show net income by year", log_experiment=False)

    assert result.generated_sql == stub_sql
    assert result.sql_execution_result is not None
    assert result.sql_execution_result.get("rows") == 2
    assert not result.errors

    orchestrator.close()
