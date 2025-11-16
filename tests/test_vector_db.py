"""
Test script for VectorDB class.
Run this to verify that vector_db.py is working correctly.
"""

import sys
from pathlib import Path

# Add backend to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.embeddings.vector_db import VectorDB
import numpy as np


def test_basic_operations():
    """Test basic add, search, and retrieve operations."""
    print("=" * 60)
    print("Testing VectorDB - Basic Operations")
    print("=" * 60)
    
    # Use a test directory
    test_dir = project_root / "data" / "vector_db" / "test_chroma_db""data/vector_db/test_chroma_db"
    
    try:
        # Test 1: Initialize VectorDB
        print("\n1. Testing initialization...")
        db = VectorDB(directory=test_dir)
        print("   ✅ VectorDB initialized successfully")
        
        # Test 2: Add embeddings
        print("\n2. Testing add_embeddings...")
        account_ids = [1, 2, 3, 4, 5]
        # Create dummy embeddings (384 dimensions for MiniLM)
        dimension = 384
        embeddings = [
            np.random.rand(dimension).tolist() for _ in range(len(account_ids))
        ]
        metadata = [
            {"account_name": f"Account {i}", "type": "test"}
            for i in account_ids
        ]
        
        db.add_embeddings(account_ids, embeddings, metadata)
        print(f"   ✅ Added {len(account_ids)} embeddings")
        
        # Test 3: Get embedding count
        print("\n3. Testing get_embedding_count...")
        count = db.get_embedding_count()
        assert count == len(account_ids), f"Expected {len(account_ids)}, got {count}"
        print(f"   ✅ Count: {count}")
        
        # Test 4: Get all IDs
        print("\n4. Testing get_all_ids...")
        all_ids = db.get_all_ids()
        assert set(all_ids) == set(account_ids), f"IDs don't match: {all_ids} vs {account_ids}"
        print(f"   ✅ Retrieved IDs: {sorted(all_ids)}")
        
        # Test 5: Search
        print("\n5. Testing search...")
        # Use one of the embeddings as query
        query_embedding = embeddings[0]  # Search for something similar to first embedding
        results = db.search(query_embedding, k=3)
        
        assert len(results) > 0, "Search returned no results"
        assert results[0][0] == account_ids[0], "Top result should be the query itself"
        print(f"   ✅ Search returned {len(results)} results")
        print(f"      Top result: Account ID {results[0][0]}, distance: {results[0][1]:.4f}")
        
        # Test 6: Load from disk
        print("\n6. Testing load from disk...")
        db2 = VectorDB.load(directory=test_dir)
        count2 = db2.get_embedding_count()
        assert count2 == count, "Loaded DB should have same count"
        print(f"   ✅ Loaded DB has {count2} embeddings")
        
        # Test 7: Clear
        print("\n7. Testing clear...")
        db.clear()
        count_after_clear = db.get_embedding_count()
        assert count_after_clear == 0, "After clear, count should be 0"
        print(f"   ✅ Cleared successfully (count: {count_after_clear})")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling."""
    print("\n" + "=" * 60)
    print("Testing VectorDB - Error Handling")
    print("=" * 60)
    
    test_dir = project_root / "data" / "vector_db" / "test_chroma_db"
    db = VectorDB(directory=test_dir)
    
    try:
        # Test: Mismatched IDs and embeddings
        print("\n1. Testing mismatched IDs/embeddings...")
        try:
            db.add_embeddings([1, 2], [[0.1] * 384], None)
            print("   ❌ Should have raised ValueError")
            return False
        except ValueError as e:
            print(f"   ✅ Correctly raised ValueError: {e}")
        
        # Test: Search on empty DB
        print("\n2. Testing search on empty DB...")
        db.clear()
        results = db.search([0.1] * 384, k=5)
        assert len(results) == 0, "Empty DB should return empty results"
        print("   ✅ Empty DB search handled correctly")
        
        print("\n✅ ERROR HANDLING TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR HANDLING TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🧪 Starting VectorDB Tests...\n")
    
    # Run tests
    test1_passed = test_basic_operations()
    test2_passed = test_error_handling()
    
    # Summary
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED! VectorDB is working correctly.")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED. Please fix the issues above.")
        sys.exit(1)