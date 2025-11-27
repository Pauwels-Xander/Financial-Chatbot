import json

import pandas as pd
import pytest

from backend.utils.answer_generator import AnswerGenerator


def test_build_numeric_llm_prompt_adds_trend_and_change():
    generator = AnswerGenerator()
    rows = [
        {"year": 2022, "revenue": 1000},
        {"year": 2023, "revenue": 1100},
    ]

    prompt = generator.build_numeric_llm_prompt("Show revenue by year", rows)

    assert '"metric_key": "revenue"' in prompt
    assert '"period_column": "year"' in prompt
    assert '"trend": "up"' in prompt
    assert '"percent_change": 10.0' in prompt
    assert "value, trend, percent_change, summary" in prompt


def test_prompt_handles_single_row_without_trend():
    generator = AnswerGenerator()
    rows = [{"period": "2023", "amount": 5000}]

    prompt = generator.build_numeric_llm_prompt("Show amount", rows)

    assert '"metric_key": "amount"' in prompt
    assert '"trend": "n/a"' in prompt
    assert '"percent_change": null' in prompt


def test_prompt_handles_zero_previous_value():
    generator = AnswerGenerator()
    rows = [
        {"year": 2022, "revenue": 0},
        {"year": 2023, "revenue": 100},
    ]

    prompt = generator.build_numeric_llm_prompt("Show revenue by year", rows)

    assert '"trend": "up"' in prompt
    assert '"previous_value": 0.0' in prompt
    assert '"percent_change": null' in prompt  # avoid divide-by-zero blowups


def test_prompt_handles_flat_trend():
    generator = AnswerGenerator()
    rows = [
        {"year": 2022, "revenue": 100},
        {"year": 2023, "revenue": 100},
    ]

    prompt = generator.build_numeric_llm_prompt("Show revenue by year", rows)

    assert '"trend": "flat"' in prompt
    assert '"percent_change": 0.0' in prompt


def test_prompt_handles_down_trend():
    generator = AnswerGenerator()
    rows = [
        {"year": 2022, "revenue": 100},
        {"year": 2023, "revenue": 90},
    ]

    prompt = generator.build_numeric_llm_prompt("Show revenue by year", rows)

    assert '"trend": "down"' in prompt
    assert '"percent_change": -10.0' in prompt


def test_prompt_without_period_column_still_heads_metric():
    generator = AnswerGenerator()
    rows = [
        {"amount": 10},
        {"amount": 20},
    ]

    prompt = generator.build_numeric_llm_prompt("Show total", rows)

    assert '"period_column": null' in prompt
    assert '"metric_key": "amount"' in prompt
    assert '"trend": "up"' in prompt


def test_prompt_supports_dataframe_and_json_inputs():
    generator = AnswerGenerator()
    df = pd.DataFrame([{"year": 2022, "revenue": 50}, {"year": 2023, "revenue": 60}])
    json_rows = json.dumps([{"period": "2022", "amount": 1}, {"period": "2023", "amount": 2}])

    prompt_from_df = generator.build_numeric_llm_prompt("Revenue", df)
    prompt_from_json = generator.build_numeric_llm_prompt("Amount", json_rows)

    assert '"metric_key": "revenue"' in prompt_from_df
    assert '"metric_key": "amount"' in prompt_from_json
    assert '"trend": "up"' in prompt_from_json


def test_prompt_raises_on_non_dict_rows():
    generator = AnswerGenerator()
    rows = [1, 2, 3]

    with pytest.raises(ValueError):
        generator.build_numeric_llm_prompt("Invalid rows", rows)


def test_generate_prefers_metric_over_period_in_time_answer():
    generator = AnswerGenerator()
    rows = [{"year": 2023, "amount": -228025728.0}]

    answer = generator.generate(
        "What was the total amount in 2023?",
        rows,
        query_metadata={"time_expressions": ["2023"]},
    )

    # Should headline the metric, not the period column
    assert "amount was" in answer
    assert "year was" not in answer
