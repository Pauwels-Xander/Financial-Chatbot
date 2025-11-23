import pytest

from backend.text_to_sql import PicardValidator, TableSchema, TextToSQLGenerator


def _generator():
    # Avoid loading the HF model; we only need the helper methods.
    return object.__new__(TextToSQLGenerator)


def test_finance_fallback_revenue_by_year():
    tables = [
        TableSchema(
            name="financials",
            columns=[
                "Current_Company_Name",
                "Year",
                "Power_Revenue_USofA_4006_4076",
                "Net_Income_Total",
            ],
        )
    ]

    sql = TextToSQLGenerator._fallback_sql(_generator(), "Show revenue by year", tables)

    assert "Power_Revenue_USofA_4006_4076" in sql
    assert "financials" in sql
    assert "ORDER BY" in sql


def test_finance_fallback_net_income_with_year_filter_and_total():
    tables = [
        TableSchema(
            name="financials",
            columns=[
                "Current_Company_Name",
                "Year",
                "Net_Income_Total",
                "Cash_and_Cash_Equivalents_USofA_1005_1070",
            ],
        )
    ]

    sql = TextToSQLGenerator._fallback_sql(
        _generator(), "total net income for 2022", tables
    )

    assert "Net_Income_Total" in sql
    assert "SUM" in sql  # total implies aggregation
    assert "2022" in sql
    assert sql.strip().endswith(";")


def test_picard_validator_handles_alias_and_unqualified_columns():
    schema = [TableSchema(name="financials", columns=["Year", "Net_Income_Total"])]
    validator = PicardValidator(schema)

    validator.validate('SELECT Year, Net_Income_Total FROM financials f WHERE f.Year = 2023;')
    validator.validate('SELECT Year, Net_Income_Total FROM financials WHERE Year = 2023;')
