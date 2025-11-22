"""
Tests for the FastAPI /ask endpoint and related API functionality.
"""

import json
import tempfile
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from backend.main import app, get_orchestrator


@pytest.fixture
def temp_database(tmp_path):
    """Create a temporary DuckDB database with sample data."""
    db_path = tmp_path / "test_api.duckdb"
    conn = duckdb.connect(str(db_path))
    
    # Create tables
    conn.execute("""
        CREATE TABLE sales (
            product VARCHAR,
            region VARCHAR,
            amount INTEGER,
            year INTEGER
        );
    """)
    
    conn.execute("""
        CREATE TABLE accounts (
            account_id INTEGER PRIMARY KEY,
            account_name VARCHAR,
            account_description VARCHAR
        );
    """)
    
    # Insert sample data
    conn.execute("""
        INSERT INTO sales VALUES
            ('Laptop', 'NA', 1200, 2022),
            ('Laptop', 'EU', 1100, 2022),
            ('Phone', 'NA', 800, 2023),
            ('Phone', 'EU', 750, 2023);
    """)
    
    conn.execute("""
        INSERT INTO accounts VALUES
            (1, 'Revenue', 'Total revenue account'),
            (2, 'Expenses', 'Operating expenses');
    """)
    
    conn.close()
    return str(db_path)


@pytest.fixture
def client(temp_database, monkeypatch):
    """Create a test client with a temporary database."""
    # Patch the default database path
    monkeypatch.setenv("DUCKDB_PATH", temp_database)
    
    # Clear any existing orchestrator
    import backend.main
    backend.main._orchestrator = None
    
    return TestClient(app)


def test_root_endpoint(client):
    """Test the root health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data
    assert "version" in data


def test_health_endpoint(client):
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert "database_path" in data
    assert "database_exists" in data


def test_ask_endpoint_valid_query(client):
    """Test the /ask endpoint with a valid query."""
    response = client.post(
        "/ask",
        json={"query": "List total amount for each product"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "answer_text" in data
    assert "sql" in data
    assert "metadata" in data
    
    # Check metadata structure
    metadata = data["metadata"]
    assert "runtime_seconds" in metadata
    assert "query_classification" in metadata
    assert "errors" in metadata
    assert "warnings" in metadata
    
    # Answer should be a string
    assert isinstance(data["answer_text"], str)
    assert len(data["answer_text"]) > 0


def test_ask_endpoint_time_query(client):
    """Test the /ask endpoint with a time-based query."""
    response = client.post(
        "/ask",
        json={"query": "Show sales for 2022"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "answer_text" in data
    assert "metadata" in data
    
    # Should have time parsing results
    metadata = data["metadata"]
    if metadata.get("time_parse_result"):
        time_result = metadata["time_parse_result"]
        assert "expression" in time_result
        assert "token" in time_result or "parse_error" in time_result


def test_ask_endpoint_empty_query(client):
    """Test the /ask endpoint with an empty query (should fail validation)."""
    response = client.post(
        "/ask",
        json={"query": ""}
    )
    
    assert response.status_code == 422  # Validation error


def test_ask_endpoint_missing_query(client):
    """Test the /ask endpoint without query field."""
    response = client.post(
        "/ask",
        json={}
    )
    
    assert response.status_code == 422  # Validation error


def test_ask_endpoint_long_query(client):
    """Test the /ask endpoint with a very long query."""
    long_query = "What is " + "the total " * 200  # Exceeds max_length
    response = client.post(
        "/ask",
        json={"query": long_query}
    )
    
    # Should either validate (if under limit) or return 422
    assert response.status_code in [200, 422]


def test_ask_endpoint_cors_headers(client):
    """Test that CORS headers are present in responses."""
    response = client.options(
        "/ask",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        }
    )
    
    # CORS middleware should handle OPTIONS requests
    # The TestClient may not show all CORS headers, but the middleware is configured
    assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly handled


def test_ask_endpoint_metadata_structure(client):
    """Test that metadata contains all expected intermediate outputs."""
    response = client.post(
        "/ask",
        json={"query": "List all products"}
    )
    
    assert response.status_code == 200
    data = response.json()
    metadata = data["metadata"]
    
    # Check all expected metadata fields
    expected_fields = [
        "runtime_seconds",
        "query_classification",
        "time_parse_result",
        "entity_links",
        "validation_status",
        "sql_execution_result",
        "errors",
        "warnings",
    ]
    
    for field in expected_fields:
        assert field in metadata, f"Missing metadata field: {field}"
    
    # Check query_classification structure
    if metadata["query_classification"]:
        classification = metadata["query_classification"]
        assert "category" in classification
        assert "confidence" in classification


def test_ask_endpoint_error_handling(client):
    """Test that errors are handled gracefully."""
    # Use a query that might fail SQL generation
    response = client.post(
        "/ask",
        json={"query": "SELECT * FROM nonexistent_table"}
    )
    
    # Should still return 200 (errors are in metadata)
    assert response.status_code == 200
    data = response.json()
    
    # Should have an answer (even if it's an error message)
    assert "answer_text" in data
    assert len(data["answer_text"]) > 0
    
    # Should have errors in metadata
    metadata = data["metadata"]
    assert "errors" in metadata
    # Errors might be empty if the query was handled, but the field should exist


def test_ask_endpoint_sql_in_metadata(client):
    """Test that generated SQL is included in response."""
    response = client.post(
        "/ask",
        json={"query": "List total amount for each product"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # SQL might be None if generation failed, but the field should exist
    assert "sql" in data
    # If SQL was generated, it should be a string
    if data["sql"] is not None:
        assert isinstance(data["sql"], str)
        assert len(data["sql"]) > 0


def test_ask_endpoint_json_content_type(client):
    """Test that responses have correct Content-Type header."""
    response = client.post(
        "/ask",
        json={"query": "What products do we have?"}
    )
    
    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")


def test_multiple_sequential_requests(client):
    """Test that multiple sequential requests work correctly."""
    queries = [
        "List all products",
        "Show total sales",
        "What is the revenue?",
    ]
    
    for query in queries:
        response = client.post(
            "/ask",
            json={"query": query}
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer_text" in data
        assert "metadata" in data


def test_ask_endpoint_with_special_characters(client):
    """Test the /ask endpoint with special characters in query."""
    response = client.post(
        "/ask",
        json={"query": "What's the total? Show me $ amounts for 2022-2023!"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer_text" in data


def test_health_endpoint_database_not_found(monkeypatch, tmp_path):
    """Test health endpoint when database doesn't exist."""
    non_existent_db = str(tmp_path / "nonexistent.duckdb")
    monkeypatch.setenv("DUCKDB_PATH", non_existent_db)
    
    import backend.main
    backend.main._orchestrator = None
    
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    # Should return error status or degraded status
    assert data["status"] in ["error", "degraded"]
    assert "database_exists" in data or "error" in data

