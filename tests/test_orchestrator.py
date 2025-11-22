"""
Tests for the pipeline orchestrator integration.
"""

import json
import tempfile
from pathlib import Path

import duckdb
import pytest

from backend.orchestrator import PipelineOrchestrator, PipelineResult


@pytest.fixture
def temp_database(tmp_path):
    """Create a temporary DuckDB database with sample data."""
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    
    # Create tables
    conn.execute("""
        CREATE TABLE accounts (
            account_id INTEGER PRIMARY KEY,
            account_name VARCHAR,
            account_description VARCHAR
        );
    """)
    
    conn.execute("""
        CREATE TABLE sales (
            product VARCHAR,
            region VARCHAR,
            amount INTEGER,
            year INTEGER
        );
    """)
    
    # Insert sample data
    conn.execute("""
        INSERT INTO accounts VALUES
            (1, 'Revenue', 'Total revenue account'),
            (2, 'Expenses', 'Operating expenses'),
            (3, 'Assets', 'Company assets');
    """)
    
    conn.execute("""
        INSERT INTO sales VALUES
            ('Laptop', 'NA', 1200, 2022),
            ('Laptop', 'EU', 1100, 2022),
            ('Phone', 'NA', 800, 2023),
            ('Phone', 'EU', 750, 2023);
    """)
    
    conn.close()
    return str(db_path)


def test_orchestrator_processes_query(temp_database):
    """Test that the orchestrator processes a query end-to-end."""
    orchestrator = PipelineOrchestrator(temp_database)
    
    result = orchestrator.process_query("List total amount for each product", log_experiment=False)
    
    assert isinstance(result, PipelineResult)
    assert result.query == "List total amount for each product"
    assert result.database_path == temp_database
    assert result.runtime_seconds > 0
    
    # Should have classification
    assert result.query_classification is not None
    assert "category" in result.query_classification
    
    # SQL generation may fail (text-to-SQL model limitations)
    # But the orchestrator should still return a result
    assert result.validation_status is not None
    
    # Should have an answer (even if it's an error message)
    assert result.answer is not None
    assert len(result.answer) > 0
    
    orchestrator.close()


def test_orchestrator_handles_time_expressions(temp_database):
    """Test that the orchestrator extracts time expressions."""
    orchestrator = PipelineOrchestrator(temp_database)
    
    result = orchestrator.process_query("Show sales for 2022 Q3", log_experiment=False)
    
    assert isinstance(result, PipelineResult)
    assert result.time_parse_result is not None
    assert "token" in result.time_parse_result or "parse_error" in result.time_parse_result
    
    orchestrator.close()


def test_orchestrator_returns_structured_json(temp_database):
    """Test that the orchestrator returns structured JSON."""
    orchestrator = PipelineOrchestrator(temp_database)
    
    result = orchestrator.process_query("What products do we have?", log_experiment=False)
    
    # Convert to dict and serialize to JSON
    result_dict = result.to_dict()
    json_str = json.dumps(result_dict)
    
    # Should be valid JSON
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)
    assert "query" in parsed
    assert "answer" in parsed
    assert "intermediate_outputs" in parsed or "generated_sql" in parsed
    
    orchestrator.close()


def test_orchestrator_handles_errors_gracefully(temp_database):
    """Test that the orchestrator handles errors gracefully."""
    orchestrator = PipelineOrchestrator(temp_database)
    
    # Query with invalid syntax/table
    result = orchestrator.process_query("SELECT * FROM nonexistent_table", log_experiment=False)
    
    assert isinstance(result, PipelineResult)
    assert result.errors is not None
    # Should still return a result (even if it's an error message)
    assert result.answer is not None
    
    orchestrator.close()

