"""
High-level interface for semantic account linking using embeddings.

This module provides the AccountLinker class which supports:
- Exact match and synonym matching via lexicon
- Semantic matching via embeddings
- Combined lexical + semantic matching (link aggregator)
"""

from typing import List, Tuple, Optional, Dict, Any
import json
from pathlib import Path
import numpy as np
import sys
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - optional dependency
    BM25Okapi = None  # type: ignore

ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # project root
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from backend.embeddings.vector_db import VectorDB


class AccountLinker:
    """
    High-level interface for semantic account linking.
    
    Implements NLP-14, NLP-15, NLP-16, NLP-17:
    - NLP-14: Embeds account_name and account_description using MiniLM
    - NLP-15: Query service returning top-k similar accounts with similarity scores
    - NLP-16: Dictionary mapping abbreviations/variants to canonical account names
    - NLP-17: Link aggregator combining lexical (exact/synonym) and semantic matches
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        vector_db_path: str = "data/vector_db/chroma_db",
        account_lexicon_path: str = "data/account_lexicon.json",
        duckdb_path: Optional[str] = None
    ):
        """
        Initialize the account linker.
        
        Args:
            model_name: Name of the sentence transformer model
            vector_db_path: Path to the ChromaDB persistence directory
            account_lexicon_path: Path to account lexicon JSON file (maps variants to canonical names)
            duckdb_path: Path to DuckDB database (stored for exact match queries)
        """
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dimension}")
        
        # Initialize vector database
        self.vector_db = VectorDB(directory=vector_db_path)
        
        # Load account lexicon (maps variants/abbreviations to canonical names)
        self.account_lexicon = self._load_account_lexicon(account_lexicon_path)
        
        # Store DuckDB path for exact match queries
        self.duckdb_path = duckdb_path
        
        # Build reverse lookup: canonical_name -> list of variants
        self._canonical_to_variants = self._build_reverse_lookup()

        # Placeholders for lexical indices (TF-IDF / BM25)
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None  # scipy sparse matrix
        self._bm25_model = None
        self._bm25_corpus: Optional[List[List[str]]] = None
        self._lexical_accounts: List[int] = []
        self._lexical_account_names: List[str] = []
    
    def _load_account_lexicon(self, lexicon_path: str) -> Dict[str, str]:
        """
        Load account lexicon from JSON file.
        
        Format: {variant/abbreviation: canonical_account_name}
        Example: {"ar": "customer accounts receivable", "sales rev": "commercial energy sales"}
        
        Returns:
            Dictionary mapping variants to canonical names
        """
        lexicon_file = Path(lexicon_path)
        if lexicon_file.exists():
            with open(lexicon_file, 'r', encoding='utf-8') as f:
                lexicon = json.load(f)
                # Normalize keys to lowercase
                return {k.lower().strip(): v for k, v in lexicon.items()}
        else:
            print(f"Warning: Account lexicon not found at {lexicon_path}. Using empty lexicon.")
            return {}
    
    def _build_reverse_lookup(self) -> Dict[str, List[str]]:
        """Build reverse lookup: canonical_name -> list of variants."""
        reverse = {}
        for variant, canonical in self.account_lexicon.items():
            canonical_lower = canonical.lower()
            if canonical_lower not in reverse:
                reverse[canonical_lower] = []
            reverse[canonical_lower].append(variant)
        return reverse
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        
        valid_texts = [str(text).strip() if text is not None and str(text).strip() else "" 
                      for text in texts]
        valid_texts = [t for t in valid_texts if t]
        
        if not valid_texts:
            return []
        
        embeddings = self.model.encode(
            valid_texts,
            convert_to_numpy=True,
            show_progress_bar=len(valid_texts) > 100,
            normalize_embeddings=False
        )
        
        return embeddings.tolist()
    
    def _find_exact_match(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Find exact match for account name (case-insensitive).
        
        Args:
            query: User query string
            
        Returns:
            Dict with account_number and confidence if exact match found, None otherwise
        """
        if not self.duckdb_path:
            return None
        
        import duckdb
        
        query_lower = query.lower().strip()
        
        try:
            con = duckdb.connect(self.duckdb_path)
            # Check account_name
            result = con.execute("""
                SELECT account_id, account_name
                FROM accounts
                WHERE LOWER(TRIM(account_name)) = ?
                LIMIT 1
            """, [query_lower]).fetchone()
            
            if not result:
                # Check account_description
                result = con.execute("""
                    SELECT account_id, account_name
                    FROM accounts
                    WHERE account_description IS NOT NULL
                      AND LOWER(TRIM(account_description)) = ?
                    LIMIT 1
                """, [query_lower]).fetchone()
            
            con.close()
            
            if result:
                return {
                    'account_number': result[0],
                    'confidence': 1.0,  # Exact match = 100% confidence
                    'match_type': 'exact',
                    'account_name': result[1]
                }
        except Exception as e:
            print(f"Warning: Could not query DuckDB for exact match: {e}")
        
        return None
    
    def _find_synonym_matches(self, query: str) -> List[Dict[str, Any]]:
        """
        Find matches from account.
        
        Args:
            query: User query string
            
        Returns:
            List of dicts with account_number, confidence, match_type
        """
        query_lower = query.lower().strip()
        matches = []
        
        # Check if query matches any variant in lexicon
        if query_lower in self.account_lexicon:
            canonical_name = self.account_lexicon[query_lower]
            # Find account(s) with this canonical name
            account_matches = self._find_accounts_by_name(canonical_name)
            for acc_id, acc_name in account_matches:
                matches.append({
                    'account_number': acc_id,
                    'confidence': 0.95,  # High confidence for exact synonym match
                    'match_type': 'synonym_exact',
                    'account_name': acc_name
                })
        else:
            # Check for partial matches
            for variant, canonical_name in self.account_lexicon.items():
                if variant in query_lower or query_lower in variant:
                    account_matches = self._find_accounts_by_name(canonical_name)
                    for acc_id, acc_name in account_matches:
                        matches.append({
                            'account_number': acc_id,
                            'confidence': 0.80,  # Medium confidence for partial match
                            'match_type': 'synonym_partial',
                            'account_name': acc_name
                        })
        
        return matches
    
    def _find_accounts_by_name(self, account_name: str) -> List[Tuple[int, str]]:
        """
        Find account IDs by account name (case-insensitive).
        
        Args:
            account_name: Canonical account name to search for
            
        Returns:
            List of (account_id, account_name) tuples
        """
        if not self.duckdb_path:
            return []
        
        import duckdb
        
        account_name_lower = account_name.lower()
        
        try:
            con = duckdb.connect(self.duckdb_path)
            results = con.execute("""
                SELECT account_id, account_name
                FROM accounts
                WHERE LOWER(TRIM(account_name)) = ?
            """, [account_name_lower]).fetchall()
            con.close()
            return results
        except Exception as e:
            print(f"Warning: Could not query DuckDB: {e}")
            return []
    
    def _distance_to_confidence(self, distance: float) -> float:
        """
        Convert vector distance to confidence score.
        
        Args:
            distance: Cosine distance from vector search
            
        Returns:
            Confidence score between 0 and 1
        """
        # Convert distance to similarity (closer = more similar)
        # For cosine distance: 0 = identical, 2 = opposite
        # Convert to confidence: 1 - (distance / 2), clamped to [0, 1]
        confidence = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
        return confidence
    
    def link_accounts(
        self,
        query_text: str,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Link aggregator: combines lexical (exact/synonym) and semantic matches.
        
        Implements NLP-17: Combines lexical matches and semantic matches,
        outputs {account_number, confidence}.
        
        Args:
            query_text: Query string to search for
            top_k: Number of top results to return
            threshold: Optional confidence threshold (0-1)
            
        Returns:
            List of dicts with format: {account_number, confidence, match_type, account_name}
        """
        if not query_text or not str(query_text).strip():
            return []
        
        query_text = str(query_text).strip()
        
        # Stage 1: Exact match (highest priority)
        exact_match = self._find_exact_match(query_text)
        if exact_match:
            return [exact_match]
        
        # Stage 2: Lexical matches (synonym/lexicon)
        lexical_matches = self._find_synonym_matches(query_text)

        # Stage 2b: TF-IDF lexical retrieval
        tfidf_matches = self._search_tfidf(query_text, top_k=top_k)

        # Stage 2c: BM25 lexical retrieval
        bm25_matches = self._search_bm25(query_text, top_k=top_k)

        # Merge lexical-style matches, keeping highest confidence per account
        merged_lexical: List[Dict[str, Any]] = []
        seen_lex: Dict[int, Dict[str, Any]] = {}
        for match in lexical_matches + tfidf_matches + bm25_matches:
            acc_num = match["account_number"]
            prev = seen_lex.get(acc_num)
            if prev is None or match["confidence"] > prev["confidence"]:
                seen_lex[acc_num] = match
        merged_lexical = list(seen_lex.values())

        # Stage 3: Semantic matches (embeddings)
        query_embedding = self.generate_embeddings([query_text])[0]
        semantic_results = self.vector_db.search(query_embedding, k=top_k * 2)  # Get more for deduplication

        # Convert semantic results to dict format
        semantic_matches: List[Dict[str, Any]] = []
        seen_accounts = set()

        # Add lexical-style matches first (they typically have higher confidence)
        for match in merged_lexical:
            acc_num = match["account_number"]
            if acc_num not in seen_accounts:
                semantic_matches.append(match)
                seen_accounts.add(acc_num)

        # Add semantic matches (avoid duplicates)
        for acc_id, distance, metadata in semantic_results:
            if metadata and metadata.get("account_id"):
                acc_num = metadata["account_id"]
                if acc_num not in seen_accounts:
                    confidence = self._distance_to_confidence(distance)
                    semantic_matches.append({
                        "account_number": acc_num,
                        "confidence": confidence,
                        "match_type": "semantic",
                        "account_name": metadata.get("text", "")
                    })
                    seen_accounts.add(acc_num)

        # Sort by confidence (descending)
        semantic_matches.sort(key=lambda x: x["confidence"], reverse=True)

        # Apply top_k and threshold
        results = semantic_matches[:top_k]
        if threshold is not None:
            results = [r for r in results if r["confidence"] >= threshold]

        return results
    
    def initialize_from_db(
        self,
        duckdb_path: str = "data/db/trial_balance.duckdb",
        clear_existing: bool = False
    ) -> int:
        """
        Initialize the vector database by loading from DuckDB.
        
        Implements NLP-14: Embeds account_name and account_description using MiniLM.
        Saves vectors + IDs in ChromaDB.
        
        Args:
            duckdb_path: Path to the DuckDB database file
            clear_existing: If True, clear existing embeddings before adding new ones
            
        Returns:
            Number of items indexed
        """
        import duckdb
        
        # Store DuckDB path for later queries
        self.duckdb_path = duckdb_path
        
        if clear_existing:
            print("Clearing existing embeddings...")
            self.vector_db.clear()
        else:
            # Warn if there are existing embeddings
            existing_count = self.vector_db.get_embedding_count()
            if existing_count > 0:
                print(f"⚠️  Warning: {existing_count} existing embeddings found.")
                print("   New embeddings will be added without clearing (may cause duplicates).")
                print("   Consider using clear_existing=True to avoid conflicts.")
        
        print(f"Connecting to DuckDB: {duckdb_path}")
        con = duckdb.connect(duckdb_path)
        
        # Get all accounts with account_name and account_description
        accounts_query = """
        SELECT DISTINCT account_id, account_name, account_description
        FROM accounts
        WHERE account_name IS NOT NULL
          AND account_name != ''
        ORDER BY account_id
        """
        
        accounts = con.execute(accounts_query).fetchall()
        
        if not accounts:
            raise ValueError(f"No accounts found in DuckDB at {duckdb_path}")
        
        print(f"Found {len(accounts)} accounts in DuckDB")

        # Build lexical indices (TF-IDF / BM25) from accounts
        self._build_lexical_index(accounts)
        
        # Prepare items to index
        items_to_index = []
        
        # Index account_name and account_description separately (NLP-14)
        for account_id, account_name, account_description in accounts:
            # Index account_name
            items_to_index.append({
                'id': f"account_name_{account_id}",
                'text': account_name,
                'type': 'account_name',
                'account_id': account_id
            })
            
            # Index account_description if it exists
            if account_description and str(account_description).strip():
                items_to_index.append({
                    'id': f"account_desc_{account_id}",
                    'text': str(account_description).strip(),
                    'type': 'account_description',
                    'account_id': account_id
                })
        
        print(f"Indexing {len(items_to_index)} items (account names and descriptions)...")
        
        # Generate embeddings
        texts = [item['text'] for item in items_to_index]
        embeddings = self.generate_embeddings(texts)
        
        # Prepare metadata
        ids = [item['id'] for item in items_to_index]
        metadata = []
        for item in items_to_index:
            meta_dict = {
                'type': item['type'],
                'text': item['text'],
                'account_id': item['account_id']
            }
            metadata.append(meta_dict)
        
        # Add to vector database
        # Convert string IDs to integers for VectorDB (it will convert back to strings for ChromaDB)
        # Using string IDs ensures uniqueness: "account_name_1", "account_desc_1", etc.
        numeric_ids = list(range(len(ids)))
        self.vector_db.add_embeddings(numeric_ids, embeddings, metadata)
        
        # Note: ChromaDB will use string IDs internally, but VectorDB interface expects integers
        # The actual ChromaDB IDs will be the string versions from the 'ids' list
        
        con.close()
        
        total_count = self.vector_db.get_embedding_count()
        print(f"✅ Vector database now contains {total_count} items.")
        
        return total_count

    def _build_lexical_index(
        self,
        accounts: List[Tuple[int, str, Optional[str]]],
    ) -> None:
        """Build TF-IDF and BM25 indices from account names/descriptions."""
        if not accounts:
            self._tfidf_vectorizer = None
            self._tfidf_matrix = None
            self._bm25_model = None
            self._bm25_corpus = None
            self._lexical_accounts = []
            self._lexical_account_names = []
            return

        texts: List[str] = []
        account_ids: List[int] = []
        account_names: List[str] = []

        for account_id, account_name, account_description in accounts:
            parts: List[str] = []
            if account_name:
                parts.append(str(account_name))
            if account_description:
                desc = str(account_description).strip()
                if desc:
                    parts.append(desc)
            full_text = " ".join(parts).strip()
            if not full_text:
                continue
            texts.append(full_text)
            account_ids.append(int(account_id))
            account_names.append(str(account_name))

        if not texts:
            return

        # Store mapping for later
        self._lexical_accounts = account_ids
        self._lexical_account_names = account_names

        # TF-IDF index
        self._tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)

        # BM25 index (optional dependency)
        if BM25Okapi is not None:
            tokenized_corpus = [t.lower().split() for t in texts]
            self._bm25_corpus = tokenized_corpus
            self._bm25_model = BM25Okapi(tokenized_corpus)
        else:
            self._bm25_corpus = None
            self._bm25_model = None

    def _search_tfidf(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """TF-IDF cosine-style retrieval over account texts."""
        if (
            not query_text
            or self._tfidf_vectorizer is None
            or self._tfidf_matrix is None
            or not self._lexical_accounts
        ):
            return []

        q_vec = self._tfidf_vectorizer.transform([query_text])
        scores = (self._tfidf_matrix @ q_vec.T).toarray().ravel()
        if scores.size == 0 or scores.max() <= 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        max_score = float(scores[top_indices[0]])
        results: List[Dict[str, Any]] = []

        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            confidence = score / max_score if max_score > 0 else 0.0
            results.append(
                {
                    "account_number": self._lexical_accounts[idx],
                    "confidence": confidence,
                    "match_type": "tfidf",
                    "account_name": self._lexical_account_names[idx],
                }
            )

        return results

    def _search_bm25(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """BM25 retrieval over account texts (if rank_bm25 is installed)."""
        if (
            not query_text
            or self._bm25_model is None
            or not self._bm25_corpus
            or not self._lexical_accounts
        ):
            return []

        tokens = query_text.lower().split()
        scores = np.array(self._bm25_model.get_scores(tokens))
        if scores.size == 0 or scores.max() <= 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        max_score = float(scores[top_indices[0]])
        results: List[Dict[str, Any]] = []

        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            confidence = score / max_score if max_score > 0 else 0.0
            results.append(
                {
                    "account_number": self._lexical_accounts[idx],
                    "confidence": confidence,
                    "match_type": "bm25",
                    "account_name": self._lexical_account_names[idx],
                }
            )

        return results
    
    def get_all_account_ids(self) -> List[int]:
        """Get all account IDs stored in the vector database."""
        all_ids = self.vector_db.get_all_ids()
        # Filter to only account types
        # This would need to check metadata - simplified for now
        return all_ids
    
    def get_embedding_count(self) -> int:
        """Get the number of embeddings stored."""
        return self.vector_db.get_embedding_count()
    
    def batch_link_accounts(
        self,
        query_texts: List[str],
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Find accounts for multiple queries at once.
        
        Args:
            query_texts: List of query strings
            top_k: Number of top results per query
            threshold: Optional confidence threshold
            
        Returns:
            List of result lists, each containing dicts with {account_number, confidence}
        """
        if not query_texts:
            return []
        
        all_results = []
        for query_text in query_texts:
            results = self.link_accounts(query_text, top_k=top_k, threshold=threshold)
            all_results.append(results)
        
        return all_results
