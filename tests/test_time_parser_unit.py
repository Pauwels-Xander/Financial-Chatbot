import pytest

from backend.utils.time_parser import (
    TimeParseError,
    TimeParseResult,
    parse_time_expression,
)
from datetime import date


def test_parse_explicit_quarter_success():
    result = parse_time_expression("Q2 2023")
    assert isinstance(result, TimeParseResult)
    assert result.granularity == "quarter"
    assert result.to_token() == "quarter:2023-Q2"


def test_parse_relative_with_base_date():
    base = date(2024, 5, 1)
    result = parse_time_expression("last quarter", base_date=base)
    assert result.start_date.year == 2024 or result.start_date.year == 2023


def test_parse_invalid_expression_raises():
    with pytest.raises(TimeParseError):
        parse_time_expression("someday maybe")





