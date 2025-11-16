"""
Script to generate embeddings for semantic layer:
- Account names (from accounts.account_name)
- Account descriptions (from accounts.account_description)

This script generates and stores embeddings in the vector database.
Implements NLP-14: Uses MiniLM to embed account_name and account_description.

For testing search functionality, use test_semantic_search.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.embeddings.linker import AccountLinker


def main():
    """Generate embeddings and build vector database."""
    
    duckdb_path = "data/db/trial_balance.duckdb"
    vector_db_path = "data/vector_db/chroma_db"
    model_name = "all-MiniLM-L6-v2"
    account_lexicon_path = "data/account_lexicon.json"
    
    print("=" * 70)
    print("Semantic Layer Embedding Generation")
    print("=" * 70)
    print(f"DuckDB path: {duckdb_path}")
    print(f"Vector DB path: {vector_db_path}")
    print(f"Account lexicon: {account_lexicon_path}")
    print(f"Model: {model_name}")
    print()
    
    if not Path(duckdb_path).exists():
        print(f"❌ Error: DuckDB file not found at {duckdb_path}")
        sys.exit(1)
    
    try:
        linker = AccountLinker(
            model_name=model_name,
            vector_db_path=vector_db_path,
            account_lexicon_path=account_lexicon_path,
            duckdb_path=duckdb_path
        )
        
        count = linker.initialize_from_db(
            duckdb_path=duckdb_path,
            clear_existing=True
        )
        
        print()
        print("=" * 70)
        print(f"✅ Successfully indexed {count} items!")
        print(f"Vector database saved to: {vector_db_path}")
        print("=" * 70)
        print()
        print("💡 Tip: Run test_semantic_search.py to test search functionality")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
