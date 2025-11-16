"""
Quick test script for account search functionality.

Usage:
1. First, generate embeddings: python backend/embeddings/generate_embeddings.py
2. Then run this script: python test_search.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.embeddings.linker import AccountLinker


def test_search():
    """Test the account search with various queries."""
    
    print("=" * 70)
    print("Account Search Test")
    print("=" * 70)
    print()
    
    # Initialize linker
    duckdb_path = "data/db/trial_balance.duckdb"
    
    if not Path(duckdb_path).exists():
        print(f"❌ Error: DuckDB file not found at {duckdb_path}")
        print("   Please make sure the database exists.")
        return
    
    print("Loading AccountLinker...")
    try:
        linker = AccountLinker(
            duckdb_path=duckdb_path
        )
        print("✅ AccountLinker loaded successfully")
        print()
    except Exception as e:
        print(f"❌ Error loading AccountLinker: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check if embeddings exist
    embedding_count = linker.get_embedding_count()
    if embedding_count == 0:
        print("⚠️  Warning: No embeddings found in vector database!")
        print("   Please run: python backend/embeddings/generate_embeddings.py")
        print()
        return
    
    print(f"✅ Found {embedding_count} embeddings in vector database")
    print()
    
    # Test queries
    test_queries = [
        # Exact matches (should use lexicon)
        "ar",
        "accounts receivable",
        "cash",
        "sales rev",
        
        # Natural language queries (should use semantic search)
        "money owed by customers",
        "cash in the bank",
        "revenue from sales",
        "operating expenses",
        
        # Abbreviations from lexicon
        "a/r",
        "ap",
    ]
    
    print("=" * 70)
    print("Testing Search Queries")
    print("=" * 70)
    print()
    
    for query in test_queries:
        print(f"Query: '{query}'")
        print("-" * 70)
        
        try:
            results = linker.link_accounts(query, top_k=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    account_number = result.get('account_number', 'N/A')
                    confidence = result.get('confidence', 0.0)
                    match_type = result.get('match_type', 'N/A')
                    account_name = result.get('account_name', 'N/A')
                    
                    print(f"  {i}. Account #{account_number}: {account_name}")
                    print(f"     Confidence: {confidence:.4f} | Match: {match_type}")
            else:
                print("  No results found")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Interactive mode
    print("=" * 70)
    print("Interactive Search Mode")
    print("=" * 70)
    print("Enter queries to search (type 'quit' to exit)")
    print()
    
    while True:
        try:
            query = input("Search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            results = linker.link_accounts(query, top_k=5)
            
            if results:
                print(f"\nFound {len(results)} result(s):")
                for i, result in enumerate(results, 1):
                    account_number = result.get('account_number', 'N/A')
                    confidence = result.get('confidence', 0.0)
                    match_type = result.get('match_type', 'N/A')
                    account_name = result.get('account_name', 'N/A')
                    
                    print(f"  {i}. Account #{account_number}: {account_name}")
                    print(f"     Confidence: {confidence:.4f} | Match: {match_type}")
            else:
                print("  No results found")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()


if __name__ == "__main__":
    test_search()

