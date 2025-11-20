import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

class VectorDB:
    """
    Vector database for storing and querying financial account embeddings
    """

    def __init__(self, directory: str | Path = "data/vector_db/chroma_db"):
        """
        Args: directory: location of the vector database
        """

        #create the directory if it doesn't exist
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        persist_directory = str(self.directory)

        #create the client and collection
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                persist_directory=persist_directory,
            ),
        )
        
        #create the collection if it doesn't exist
        self.collection = self.client.get_or_create_collection(
            name="financial_accounts",
            metadata={"description": "Financial accounts embeddings"}
        )
    
    def add_embeddings(
        self,
        ids: List[int], 
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add embeddings to the vector database

        Args:
        ids: List of account ID unique to each embedding
        embeddings: List of embeddings to add
        metadata: Optional list of metadata dictionaries for each embedding
        """

        #check if the number of IDs and embeddings match
        if len(ids) != len(embeddings):
            raise ValueError(f"Number of IDs ({len(ids)}) must match number of embeddings ({len(embeddings)}")

        #convert IDs to strings and embeddings to list
        id_strings = [str(account_id) for account_id in ids]

        #check if the metadata is provided and if it matches the number of IDs
        if metadata is None:
            metadata = [{}] * len(ids)
        elif len(metadata) != len(ids):
            raise ValueError(f"Number of metadata ({len(metadata)}) must match number of IDs ({len(ids)}")
        
        #add the embeddings to the vector database
        self.collection.add(
            ids=id_strings,
            embeddings=embeddings,
            metadatas=metadata
        )

    def search(
        self, query_embeddings: List[float], k: int = 5) -> List[Tuple[int, float, Optional[Dict[str, Any]]]]:
        """
        Search for similar embeddings
        Args:
            query_embeddings: Single query embedding to search for (as list)
            k: Number of nearest neighbors to return
        Returns:
            List of tuples containing (account_id, distance, metadata), sorted by distance
        """
        # Check if database is empty
        count = self.collection.count()
        if count == 0:
            return []
        
        # Convert to 2D list format (ChromaDB expects list of lists)
        if isinstance(query_embeddings[0], (int, float)):
            # Single embedding as list
            query_list = [query_embeddings]
        else:
            # Already in list of lists format
            query_list = query_embeddings
        
        #search for nearest neighbors
        results = self.collection.query(
            query_embeddings=query_list,
            n_results=min(k, count)
        )
        
        #format the result before return
        formatted_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for acc_id_str, distance, metadata in zip(
                results["ids"][0], 
                results["distances"][0], 
                results["metadatas"][0] if results["metadatas"] and results["metadatas"][0] else [None] * len(results["ids"][0])
            ):
                acc_id = int(acc_id_str)
                formatted_results.append((acc_id, float(distance), metadata))
        return formatted_results
    
    def get_all_ids(self) -> List[int]:
        """
        Get all account IDs in the vector database
        Returns:
        List of all account IDs
        """
        return [int(id_str) for id_str in self.collection.get()["ids"]]

    def get_embedding_count(self) -> int:
        """Get the number of embeddings stored."""
        return self.collection.count()
    
    def save(self, base_path: str = None):
        """
        Save is automatic with ChromaDB PersistentClient.
        This method exists for API compatibility.
        """
        if base_path:
            Path(base_path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, directory: str = "data/vector_db/chroma_db") -> 'VectorDB':
        """
        Load existing ChromaDB database.
        Just creates a new instance pointing to the same directory.
        
        Args:
            directory: Directory where ChromaDB data is stored
            
        Returns:
            Loaded VectorDB instance
        """
        return cls(directory=directory)
    
    def clear(self):
        """Clear all embeddings and reset the database."""
        self.client.delete_collection("financial_accounts")
        self.collection = self.client.get_or_create_collection(
            name="financial_accounts",
            metadata={"description": "Financial account embeddings"}
        )
