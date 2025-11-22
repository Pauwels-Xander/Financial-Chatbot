#!/usr/bin/env python3
"""
Integration test script for the pipeline orchestrator.

This script creates a test database, runs a query through the complete pipeline,
and displays the structured results.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import duckdb

from backend.orchestrator import PipelineOrchestrator


def create_test_database(db_path: str) -> None:
    """Create a test DuckDB database with sample financial data."""
    print(f"Creating test database at {db_path}...")
    conn = duckdb.connect(db_path)
    
    # Create accounts table
    conn.execute("""
        CREATE TABLE accounts (
            account_id INTEGER PRIMARY KEY,
            account_name VARCHAR,
            account_description VARCHAR
        );
    """)
    
    # Create sales table
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
            ('Phone', 'EU', 750, 2023),
            ('Tablet', 'NA', 600, 2023);
    """)
    
    conn.close()
    print("✅ Test database created successfully")


def test_orchestrator():
    """Test the orchestrator with sample queries."""
    # Create temporary database
    import tempfile
    tmp_dir = tempfile.gettempdir()
    db_path = str(Path(tmp_dir) / f"test_orchestrator_{tempfile.gettempprefix()}.duckdb")
    # Remove if exists
    if Path(db_path).exists():
        Path(db_path).unlink()
    
    try:
        create_test_database(db_path)
        
        # Initialize orchestrator
        print("\n" + "="*60)
        print("Initializing Pipeline Orchestrator...")
        print("="*60)
        orchestrator = PipelineOrchestrator(db_path)
        
        # Test queries
        test_queries = [
            "List total amount for each product",
            "What products do we have?",
            "Show sales for 2022",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"Test Query {i}: {query}")
            print('='*60)
            
            try:
                result = orchestrator.process_query(query, log_experiment=False)
                
                # Display results
                print(f"\n📊 Query Classification:")
                if result.query_classification:
                    print(f"   Category: {result.query_classification.get('category', 'N/A')}")
                    print(f"   Confidence: {result.query_classification.get('confidence', 0.0):.2f}")
                
                print(f"\n⏰ Time Parsing:")
                if result.time_parse_result:
                    print(f"   Expression: {result.time_parse_result.get('expression', 'N/A')}")
                    if 'token' in result.time_parse_result:
                        print(f"   Token: {result.time_parse_result['token']}")
                else:
                    print("   No time expressions detected")
                
                print(f"\n🔗 Entity Links:")
                if result.entity_links:
                    for link in result.entity_links[:3]:  # Show top 3
                        print(f"   Account {link.get('account_number', 'N/A')}: {link.get('account_name', 'N/A')} (confidence: {link.get('confidence', 0.0):.2f})")
                else:
                    print("   No entity links found")
                
                print(f"\n💾 Generated SQL:")
                if result.generated_sql:
                    print(f"   {result.generated_sql}")
                    print(f"   Validation: {result.validation_status or 'N/A'}")
                else:
                    print("   ❌ SQL generation failed")
                
                print(f"\n📈 SQL Execution:")
                if result.sql_execution_result:
                    rows = result.sql_execution_result.get('rows', 0)
                    columns = result.sql_execution_result.get('columns', [])
                    print(f"   Rows: {rows}")
                    print(f"   Columns: {', '.join(columns)}")
                    if rows > 0 and rows <= 5:
                        data = result.sql_execution_result.get('data', [])
                        print(f"   Sample data:")
                        for row in data[:3]:
                            print(f"     {row}")
                else:
                    print("   ❌ SQL execution failed")
                
                print(f"\n💬 Final Answer:")
                print(f"   {result.answer or 'No answer generated'}")
                
                print(f"\n⏱️  Runtime: {result.runtime_seconds:.3f} seconds")
                
                if result.errors:
                    print(f"\n⚠️  Errors:")
                    for error in result.errors:
                        print(f"   - {error}")
                
                if result.warnings:
                    print(f"\n⚠️  Warnings:")
                    for warning in result.warnings:
                        print(f"   - {warning}")
                
                # Show JSON structure
                print(f"\n📋 Structured JSON (first 500 chars):")
                result_json = json.dumps(result.to_dict(), indent=2)
                print(result_json[:500] + "..." if len(result_json) > 500 else result_json)
                
            except Exception as exc:
                print(f"❌ Error processing query: {exc}")
                import traceback
                traceback.print_exc()
        
        orchestrator.close()
        print(f"\n{'='*60}")
        print("✅ All tests completed!")
        print('='*60)
        
    finally:
        # Clean up
        try:
            Path(db_path).unlink()
        except:
            pass


if __name__ == "__main__":
    test_orchestrator()

