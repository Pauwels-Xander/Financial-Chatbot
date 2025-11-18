from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Literal, Optional

import dateparser

Granularity = Literal["day", "month", "quarter", "year"]


class TimeParseError(ValueError):
    """Raised when an expression cannot be resolved into a known time range."""


@dataclass(frozen=True)
class TimeParseResult:
    expression: str
    granularity: Granularity
    start_date: date
    end_date: date

    def to_token(self) -> str:
        match self.granularity:
            case "day":
                return f"day:{self.start_date.isoformat()}"
            case "month":
                return f"month:{self.start_date.strftime('%Y-%m')}"
            case "quarter":
                quarter = ((self.start_date.month - 1) // 3) + 1
                return f"quarter:{self.start_date.year}-Q{quarter}"
            case "year":
                return f"year:{self.start_date.year}"
            case _:
                raise TimeParseError(f"Unsupported granularity: {self.granularity}")

    def to_dict(self) -> Dict[str, str]:
        return {
            "expression": self.expression,
            "token": self.to_token(),
            "granularity": self.granularity,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
        }


_QUARTER_EXPLICIT = re.compile(
    r"""
    ^\s*
    (?:
        (?P<year>\d{4})\s*[-/]?\s*[Qq](?P<quarter>[1-4])
        |
        [Qq](?P<quarter_alt>[1-4])\s*[-/]?\s*(?P<year_alt>\d{4})
    )
    \s*$
    """,
    re.VERBOSE,
)

_QUARTER_RELATIVE = re.compile(r"^(?P<modifier>last|this|next)\s+quarter$", re.IGNORECASE)
_YEAR_EXPLICIT = re.compile(r"^\s*(?P<year>\d{4})\s*$")
_MONTH_NAME = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)", re.IGNORECASE)


def parse_time_expression(expression: str, *, base_date: Optional[date] = None) -> TimeParseResult:
    """Interpret a natural language or structured time expression into a normalized range."""
    if not expression or not expression.strip():
        raise TimeParseError("Empty expression cannot be parsed")

    normalized = expression.strip()

    if (result := _try_quarter_expression(normalized, base_date)) is not None:
        return result

    if (result := _try_year_expression(normalized)) is not None:
        return result

    if (result := _try_month_or_day_expression(normalized, base_date)) is not None:
        return result

    raise TimeParseError(f"Unsupported time expression: '{expression}'")


def _try_quarter_expression(expression: str, base_date: Optional[date]) -> Optional[TimeParseResult]:
    match = _QUARTER_EXPLICIT.match(expression)
    if match:
        year = int(match.group("year") or match.group("year_alt"))
        quarter = int(match.group("quarter") or match.group("quarter_alt"))
        start, end = _quarter_bounds(year, quarter)
        return TimeParseResult(expression, "quarter", start, end)

    match = _QUARTER_RELATIVE.match(expression)
    if match:
        base = base_date or date.today()
        modifier = match.group("modifier").lower()
        offset = {"last": -1, "this": 0, "next": 1}[modifier]
        base_quarter = ((base.month - 1) // 3) + 1
        base_year = base.year
        target_quarter = base_quarter + offset
        target_year = base_year
        if target_quarter < 1:
            target_quarter += 4
            target_year -= 1
        elif target_quarter > 4:
            target_quarter -= 4
            target_year += 1
        start, end = _quarter_bounds(target_year, target_quarter)
        return TimeParseResult(expression, "quarter", start, end)

    return None


def _try_year_expression(expression: str) -> Optional[TimeParseResult]:
    match = _YEAR_EXPLICIT.match(expression)
    if match:
        year = int(match.group("year"))
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        return TimeParseResult(expression, "year", start, end)

    relative_years = {"last year": -1, "this year": 0, "next year": 1}
    key = expression.lower()
    if key in relative_years:
        base_year = date.today().year + relative_years[key]
        start = date(base_year, 1, 1)
        end = date(base_year, 12, 31)
        return TimeParseResult(expression, "year", start, end)

    return None


def _try_month_or_day_expression(expression: str, base_date: Optional[date]) -> Optional[TimeParseResult]:
    base_settings = {
        "RELATIVE_BASE": datetime.combine(base_date or date.today(), datetime.min.time()),
        "PREFER_DAY_OF_MONTH": "first",
        "PREFER_DATES_FROM": "past",
    }

    parsed = dateparser.parse(
        expression,
        settings={**base_settings, "STRICT_PARSING": True},
    ) or dateparser.parse(
        expression,
        settings={**base_settings, "STRICT_PARSING": False},
    )
    if not parsed:
        return None

    if _MONTH_NAME.search(expression):
        start = date(parsed.year, parsed.month, 1)
        end = _end_of_month(start)
        return TimeParseResult(expression, "month", start, end)

    if re.search(r"\b\d{1,2}(st|nd|rd|th)?\b", expression):
        the_date = parsed.date()
        return TimeParseResult(expression, "day", the_date, the_date)

    start = date(parsed.year, parsed.month, 1)
    end = _end_of_month(start)
    return TimeParseResult(expression, "month", start, end)


def _quarter_bounds(year: int, quarter: int) -> tuple[date, date]:
    if quarter not in {1, 2, 3, 4}:
        raise TimeParseError(f"Quarter must be 1-4, received: {quarter}")

    start_month = 3 * (quarter - 1) + 1
    start = date(year, start_month, 1)
    end_month_start = _add_months(start, 2)
    end = _end_of_month(end_month_start)
    return start, end


def _add_months(value: date, months: int) -> date:
    month = value.month - 1 + months
    year = value.year + month // 12
    month = month % 12 + 1
    return date(year, month, 1)


def _end_of_month(value: date) -> date:
    next_month = _add_months(value, 1)
    return next_month - timedelta(days=1)


__all__ = ["parse_time_expression", "TimeParseResult", "TimeParseError"]
