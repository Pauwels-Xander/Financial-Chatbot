from datetime import date

import pandas as pd  # noqa: F401 - placeholder for other utils tests
import pytest

from backend.utils.time_parser import TimeParseError, parse_time_expression


def test_parse_explicit_quarter():
    result = parse_time_expression("2022 Q3")
    assert result.granularity == "quarter"
    assert result.start_date.isoformat() == "2022-07-01"
    assert result.end_date.isoformat() == "2022-09-30"
    assert result.to_token() == "quarter:2022-Q3"


def test_parse_relative_quarter_with_base_date():
    base = date(2023, 2, 10)
    result = parse_time_expression("last quarter", base_date=base)
    assert result.start_date.isoformat() == "2022-10-01"
    assert result.end_date.isoformat() == "2022-12-31"


def test_parse_month_expression():
    base = date(2024, 6, 15)
    result = parse_time_expression("March 2023", base_date=base)
    assert result.granularity == "month"
    assert result.start_date.isoformat() == "2023-03-01"
    assert result.end_date.isoformat() == "2023-03-31"
    assert result.to_token() == "month:2023-03"


def test_parse_unknown_expression_raises():
    with pytest.raises(TimeParseError):
        parse_time_expression("the distant future")
