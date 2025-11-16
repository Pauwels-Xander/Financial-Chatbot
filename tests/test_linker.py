"""
Test script for AccountLinker class.
Run this to verify that linker.py is working correctly.
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.embeddings.linker import AccountLinker
from backend.embeddings.vector_db import VectorDB
import numpy as np


def test_initialization():
    """Test AccountLinker initialization."""
    print("=" * 60)
    print("Testing AccountLinker - Initialization")
    print("=" * 60)
    
    try:
        # Use a test directory
        test_dir = project_root / "data" / "vector_db" / "test_linker_chroma_db"
        test_dir_str = str(test_dir)
        
        print("\n1. Testing initialization...")
        linker = AccountLinker(
            model_name="all-MiniLM-L6-v2",
            vector_db_path=test_dir_str
        )
        
        assert linker.model is not None, "Model should be loaded"
        assert linker.embedding_dimension == 384, f"Expected dimension 384, got {linker.embedding_dimension}"
        assert linker.vector_db is not None, "VectorDB should be initialized"
        print("   ✅ AccountLinker initialized successfully")
        
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generate_embeddings():
    """Test embedding generation."""
    print("\n" + "=" * 60)
    print("Testing AccountLinker - Generate Embeddings")
    print("=" * 60)
    
    test_dir = project_root / "data" / "vector_db" / "test_linker_chroma_db"
    test_dir_str = str(test_dir)
    
    try:
        linker = AccountLinker(vector_db_path=test_dir_str)
        
        # Test 1: Normal text
        print("\n1. Testing normal text embedding...")
        texts = ["revenue", "expenses", "assets"]
        embeddings = linker.generate_embeddings(texts)
        
        assert len(embeddings) == len(texts), f"Expected {len(texts)} embeddings, got {len(embeddings)}"
        assert len(embeddings[0]) == 384, f"Expected dimension 384, got {len(embeddings[0])}"
        assert all(isinstance(emb, list) for emb in embeddings), "Embeddings should be lists"
        print(f"   ✅ Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
        
        # Test 2: Empty list
        print("\n2. Testing empty list...")
        empty_embeddings = linker.generate_embeddings([])
        assert empty_embeddings == [], "Empty list should return empty list"
        print("   ✅ Empty list handled correctly")
        
        # Test 3: None/empty strings
        print("\n3. Testing None/empty strings...")
        mixed_texts = ["valid text", None, "", "   ", "another valid"]
        mixed_embeddings = linker.generate_embeddings(mixed_texts)
        # Should only return embeddings for valid texts
        assert len(mixed_embeddings) > 0, "Should have some valid embeddings"
        print(f"   ✅ Filtered invalid texts, got {len(mixed_embeddings)} embeddings")
        
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_link_accounts():
    """Test account linking functionality."""
    print("\n" + "=" * 60)
    print("Testing AccountLinker - Link Accounts")
    print("=" * 60)
    
    test_dir = project_root / "data" / "vector_db" / "test_linker_chroma_db"
    test_dir_str = str(test_dir)
    
    try:
        linker = AccountLinker(vector_db_path=test_dir_str)
        
        # First, add some test accounts
        print("\n1. Setting up test accounts...")
        account_ids = [1, 2, 3, 4, 5]
        account_names = [
            "Sales Revenue",
            "Operating Expenses",
            "Cash and Cash Equivalents",
            "Accounts Receivable",
            "Long-term Debt"
        ]
        
        # Generate embeddings and add to vector DB
        embeddings = linker.generate_embeddings(account_names)
        metadata = [{"account_name": name} for name in account_names]
        linker.vector_db.add_embeddings(account_ids, embeddings, metadata)
        print(f"   ✅ Added {len(account_ids)} test accounts")
        
        # Test 2: Search for similar account
        print("\n2. Testing link_accounts with 'revenue' query...")
        results = linker.link_accounts("revenue", top_k=3)
        
        assert len(results) > 0, "Should return at least one result"
        assert len(results) <= 3, "Should return at most top_k results"
        assert all(isinstance(r, dict) for r in results), "Results should be dicts with {account_number, confidence}"
        assert 'account_number' in results[0], "Results should have account_number"
        assert 'confidence' in results[0], "Results should have confidence"
        print(f"   ✅ Found {len(results)} results")
        print(f"      Top result: Account #{results[0]['account_number']} - {results[0].get('account_name', 'N/A')} (confidence: {results[0]['confidence']:.4f})")
        
        # Test 3: Search with threshold
        print("\n3. Testing link_accounts with threshold...")
        results_with_threshold = linker.link_accounts("revenue", top_k=5, threshold=0.5)
        assert all(r['confidence'] >= 0.5 for r in results_with_threshold), "All results should be above confidence threshold"
        print(f"   ✅ Threshold filtering works ({len(results_with_threshold)} results)")
        
        # Test 4: Empty query
        print("\n4. Testing empty query...")
        empty_results = linker.link_accounts("", top_k=5)
        assert empty_results == [], "Empty query should return empty results"
        print("   ✅ Empty query handled correctly")
        
        # Test 5: Query with no matches (very specific)
        print("\n5. Testing query with no close matches...")
        no_match_results = linker.link_accounts("completely unrelated query about quantum physics", top_k=5)
        # Should still return results (just not very similar)
        assert len(no_match_results) > 0, "Should return results even if not very similar"
        print(f"   ✅ Query with no close matches handled ({len(no_match_results)} results)")
        
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_link_accounts():
    """Test batch account linking."""
    print("\n" + "=" * 60)
    print("Testing AccountLinker - Batch Link Accounts")
    print("=" * 60)
    
    test_dir = project_root / "data" / "vector_db" / "test_linker_chroma_db"
    test_dir_str = str(test_dir)
    
    try:
        linker = AccountLinker(vector_db_path=test_dir_str)
        
        # Set up test accounts
        print("\n1. Setting up test accounts...")
        account_ids = [1, 2, 3]
        account_names = ["Revenue", "Expenses", "Assets"]
        embeddings = linker.generate_embeddings(account_names)
        metadata = [{"account_name": name} for name in account_names]
        linker.vector_db.add_embeddings(account_ids, embeddings, metadata)
        
        # Test batch search
        print("\n2. Testing batch_link_accounts...")
        queries = ["revenue", "costs", "money"]
        batch_results = linker.batch_link_accounts(queries, top_k=2)
        
        assert len(batch_results) == len(queries), f"Expected {len(queries)} result lists, got {len(batch_results)}"
        assert all(isinstance(result_list, list) for result_list in batch_results), "Each result should be a list"
        assert all(len(result_list) <= 2 for result_list in batch_results), "Each result should have at most top_k items"
        assert all(isinstance(r, dict) and 'account_number' in r for result_list in batch_results for r in result_list), "Results should be dicts with account_number"
        print(f"   ✅ Batch search returned {len(batch_results)} result lists")
        for i, (query, results) in enumerate(zip(queries, batch_results)):
            print(f"      Query '{query}': {len(results)} results")
        
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_initialize_from_db():
    """Test initialization from DuckDB (with mock data)."""
    print("\n" + "=" * 60)
    print("Testing AccountLinker - Initialize from DuckDB")
    print("=" * 60)
    
    test_dir = project_root / "data" / "vector_db" / "test_linker_chroma_db"
    test_dir_str = str(test_dir)
    
    # Use a test DuckDB file in the project's test directory
    test_db_path = project_root / "tests" / "test_accounts.duckdb"
    test_db_path_str = str(test_db_path)
    
    try:
        import duckdb
        import os
        
        # Remove existing test DB if it exists
        if test_db_path.exists():
            os.unlink(test_db_path)
        
        # Create test DuckDB database
        print("\n1. Creating test DuckDB database...")
        con = duckdb.connect(test_db_path_str)
        con.execute("""
            CREATE TABLE accounts (
                account_id INTEGER PRIMARY KEY,
                account_name VARCHAR,
                account_description VARCHAR,
                is_control BOOLEAN
            )
        """)
        
        # Insert test data
        test_accounts = [
            (1, "Sales Revenue", "Revenue from sales", True),
            (2, "Operating Expenses", "Expenses for operations", True),
            (3, "Cash", "Cash and cash equivalents", True),
            (4, "Accounts Receivable", "Money owed by customers", False),
        ]
        con.executemany(
            "INSERT INTO accounts (account_id, account_name, account_description, is_control) VALUES (?, ?, ?, ?)",
            test_accounts
        )
        con.close()
        
        print(f"   ✅ Created test database with {len(test_accounts)} accounts")
        
        # Test initialization
        print("\n2. Testing initialize_from_db...")
        linker = AccountLinker(vector_db_path=test_dir_str)
        count = linker.initialize_from_db(
            duckdb_path=test_db_path_str,
            clear_existing=True
        )
        
        # Should have 2 embeddings per account (account_name + account_description)
        expected_count = len(test_accounts) * 2
        assert count == expected_count, f"Expected {expected_count} embeddings (2 per account), got {count}"
        assert linker.get_embedding_count() == expected_count, "Embedding count should match"
        print(f"   ✅ Initialized {count} embeddings from DuckDB")
        
        # Test that we can search
        print("\n3. Testing search after initialization...")
        results = linker.link_accounts("revenue", top_k=2)
        assert len(results) > 0, "Should be able to search after initialization"
        print(f"   ✅ Search works after initialization ({len(results)} results)")
        
        # Test incremental update (should skip existing)
        print("\n4. Testing incremental update...")
        count_before = linker.get_embedding_count()
        count_after = linker.initialize_from_db(
            duckdb_path=test_db_path_str,
            clear_existing=False
        )
        assert count_after == count_before, "Should not add duplicates"
        print(f"   ✅ Incremental update works (count: {count_after})")
        
        # Cleanup
        if test_db_path.exists():
            os.unlink(test_db_path)
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup on error
        try:
            if test_db_path.exists():
                os.unlink(test_db_path)
        except:
            pass
        return False


def test_helper_methods():
    """Test helper methods."""
    print("\n" + "=" * 60)
    print("Testing AccountLinker - Helper Methods")
    print("=" * 60)
    
    test_dir = project_root / "data" / "vector_db" / "test_linker_chroma_db"
    test_dir_str = str(test_dir)
    
    try:
        linker = AccountLinker(vector_db_path=test_dir_str)
        
        # Clear any existing data to ensure clean test state
        linker.vector_db.clear()
        
        # Add some test data
        account_ids = [1, 2, 3]
        account_names = ["Test Account 1", "Test Account 2", "Test Account 3"]
        embeddings = linker.generate_embeddings(account_names)
        metadata = [{"account_name": name} for name in account_names]
        linker.vector_db.add_embeddings(account_ids, embeddings, metadata)
        
        # Test get_all_account_ids
        print("\n1. Testing get_all_account_ids...")
        all_ids = linker.get_all_account_ids()
        assert set(all_ids) == set(account_ids), f"Expected {set(account_ids)}, got {set(all_ids)}"
        print(f"   ✅ Retrieved {len(all_ids)} account IDs")
        
        # Test get_embedding_count
        print("\n2. Testing get_embedding_count...")
        count = linker.get_embedding_count()
        assert count == len(account_ids), f"Expected {len(account_ids)}, got {count}"
        print(f"   ✅ Embedding count: {count}")
        
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🧪 Starting AccountLinker Tests...\n")
    
    # Run all tests
    tests = [
        ("Initialization", test_initialization),
        ("Generate Embeddings", test_generate_embeddings),
        ("Link Accounts", test_link_accounts),
        ("Batch Link Accounts", test_batch_link_accounts),
        ("Initialize from DuckDB", test_initialize_from_db),
        ("Helper Methods", test_helper_methods),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! AccountLinker is working correctly.")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED. Please fix the issues above.")
        sys.exit(1)
