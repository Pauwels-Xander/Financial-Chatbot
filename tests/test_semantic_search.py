"""
Test script to verify semantic search works for natural language queries.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.embeddings.linker import AccountLinker

def test_semantic_queries():
    """Test various natural language queries."""
    
    # Initialize linker (assumes embeddings are already generated)
    linker = AccountLinker(
        duckdb_path="data/db/trial_balance.duckdb"
    )
    
    # Test queries - natural language descriptions
    test_queries = [
        ("money owed by customer", "Should match: Accounts Receivable"),
        ("money we owe to suppliers", "Should match: Accounts Payable"),
        ("cash in bank", "Should match: Cash, Bank accounts"),
        ("money we earned from sales", "Should match: Revenue, Sales"),
        ("costs of running the business", "Should match: Operating Expenses"),
        ("things we own that have value", "Should match: Assets"),
        ("money we borrowed", "Should match: Loans, Debt"),
    ]
    
    print("=" * 70)
    print("Testing Semantic Search with Natural Language Queries")
    print("=" * 70)
    print()
    
    for query, expected in test_queries:
        print(f"Query: '{query}'")
        print(f"Expected: {expected}")
        print("Results:")
        
        results = linker.link_accounts(query, top_k=5)
        
        if results:
            for i, result in enumerate(results, 1):
                account_number = result.get('account_number', 'N/A')
                confidence = result.get('confidence', 0.0)
                match_type = result.get('match_type', 'N/A')
                account_name = result.get('account_name', 'N/A')
                
                print(f"  {i}. Account #{account_number}: {account_name}")
                print(f"      Confidence: {confidence:.4f} | Match type: {match_type}")
        else:
            print("  No results found")
        
        print()

if __name__ == "__main__":
    test_semantic_queries()
