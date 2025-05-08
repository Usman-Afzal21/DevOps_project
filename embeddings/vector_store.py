"""
Vector Store Module for Banking Transaction Data.

This module handles:
- Building a vector database from embeddings
- Managing collections for transactions and summaries
- Retrieving relevant data for the RAG pipeline
"""

import os
import pandas as pd
import numpy as np
import chromadb
import logging
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple

# Prevent ChromaDB from trying to load .env file
os.environ["CHROMA_OVERRIDE_ENV_VAR"] = "true"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TransactionVectorStore:
    """
    Build and manage vector database for transaction data retrieval.
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the vector database.
                               If None, the database will be in-memory only.
        """
        self.persist_directory = persist_directory
        
        # Configure ChromaDB
        if persist_directory:
            logger.info(f"Initializing persistent vector database at {persist_directory}")
            os.makedirs(persist_directory, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=persist_directory, 
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            logger.info("Initializing in-memory vector database")
            self.chroma_client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )
        
        # Collections for different types of data
        self.transactions_collection = None
        self.summaries_collection = None
    
    def create_transaction_collection(self, collection_name: str = "transaction_embeddings"):
        """
        Create a collection for transaction embeddings.
        
        Args:
            collection_name: Name for the collection
        """
        # Get or create collection
        try:
            self.transactions_collection = self.chroma_client.get_collection(collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except:
            logger.info(f"Creating new collection: {collection_name}")
            self.transactions_collection = self.chroma_client.create_collection(collection_name)
    
    def create_summary_collection(self, collection_name: str = "summary_embeddings"):
        """
        Create a collection for branch summary embeddings.
        
        Args:
            collection_name: Name for the collection
        """
        # Get or create collection
        try:
            self.summaries_collection = self.chroma_client.get_collection(collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except:
            logger.info(f"Creating new collection: {collection_name}")
            self.summaries_collection = self.chroma_client.create_collection(collection_name)
    
    def add_transaction_embeddings(self, 
                                   embeddings: np.ndarray, 
                                   texts: List[str], 
                                   ids: List[str], 
                                   metadata: List[Dict[str, Any]]):
        """
        Add transaction embeddings to the vector store.
        
        Args:
            embeddings: Embedding vectors
            texts: Transaction texts
            ids: Unique identifier for each embedding
            metadata: Metadata for each embedding
        """
        if self.transactions_collection is None:
            raise ValueError("Transaction collection not initialized. Call create_transaction_collection() first.")
        
        logger.info(f"Adding {len(embeddings)} transaction embeddings to vector store")
        
        # Convert embeddings to list of lists for ChromaDB
        embeddings_list = embeddings.tolist()
        
        # Add to collection
        self.transactions_collection.add(
            embeddings=embeddings_list,
            documents=texts,
            ids=ids,
            metadatas=metadata
        )
        
        logger.info(f"Added {len(embeddings)} transaction embeddings to vector store")
    
    def add_summary_embeddings(self, 
                              embeddings: np.ndarray, 
                              texts: List[str], 
                              ids: List[str], 
                              metadata: List[Dict[str, Any]]):
        """
        Add branch summary embeddings to the vector store.
        
        Args:
            embeddings: Embedding vectors
            texts: Summary texts
            ids: Unique identifier for each embedding
            metadata: Metadata for each embedding
        """
        if self.summaries_collection is None:
            raise ValueError("Summary collection not initialized. Call create_summary_collection() first.")
        
        logger.info(f"Adding {len(embeddings)} summary embeddings to vector store")
        
        # Convert embeddings to list of lists for ChromaDB
        embeddings_list = embeddings.tolist()
        
        # Add to collection
        self.summaries_collection.add(
            embeddings=embeddings_list,
            documents=texts,
            ids=ids,
            metadatas=metadata
        )
        
        logger.info(f"Added {len(embeddings)} summary embeddings to vector store")
    
    def load_from_files(self, 
                       transaction_embeddings_path: str, 
                       transaction_texts_path: str, 
                       transaction_metadata_path: str,
                       summary_embeddings_path: str, 
                       summary_texts_path: str, 
                       summary_metadata_path: str):
        """
        Load embeddings, texts, and metadata from files and add to vector store.
        
        Args:
            transaction_embeddings_path: Path to transaction embeddings file (.npy)
            transaction_texts_path: Path to transaction texts file (.txt)
            transaction_metadata_path: Path to transaction metadata file (.csv)
            summary_embeddings_path: Path to summary embeddings file (.npy)
            summary_texts_path: Path to summary texts file (.txt)
            summary_metadata_path: Path to summary metadata file (.csv)
        """
        logger.info("Loading data from files")
        
        # Load transaction data
        transaction_embeddings = np.load(transaction_embeddings_path)
        
        with open(transaction_texts_path, 'r') as f:
            transaction_texts = f.read().splitlines()
        
        transaction_metadata = pd.read_csv(transaction_metadata_path).to_dict('records')
        transaction_ids = [str(meta['id']) for meta in transaction_metadata]
        
        # Load summary data
        summary_embeddings = np.load(summary_embeddings_path)
        
        with open(summary_texts_path, 'r') as f:
            summary_texts = f.read().splitlines()
        
        summary_metadata = pd.read_csv(summary_metadata_path).to_dict('records')
        summary_ids = [str(meta['id']) for meta in summary_metadata]
        
        # Create collections
        self.create_transaction_collection()
        self.create_summary_collection()
        
        # Add to vector store
        self.add_transaction_embeddings(
            embeddings=transaction_embeddings,
            texts=transaction_texts,
            ids=transaction_ids,
            metadata=transaction_metadata
        )
        
        self.add_summary_embeddings(
            embeddings=summary_embeddings,
            texts=summary_texts,
            ids=summary_ids,
            metadata=summary_metadata
        )
        
        logger.info("Successfully loaded data into vector store")
    
    def query_transactions(self, query_text: str, n_results: int = 5, filter_criteria: Optional[Dict] = None) -> Dict:
        """
        Query the transaction collection for similar embeddings.
        
        Args:
            query_text: Text query to search for
            n_results: Number of results to return
            filter_criteria: Optional filter criteria
            
        Returns:
            Query results from ChromaDB
        """
        if self.transactions_collection is None:
            raise ValueError("Transaction collection not initialized")
        
        logger.info(f"Querying transactions with: '{query_text}'")
        
        results = self.transactions_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_criteria
        )
        
        logger.info(f"Found {len(results['documents'][0])} matching transactions")
        
        return results
    
    def query_summaries(self, query_text: str, n_results: int = 5, filter_criteria: Optional[Dict] = None) -> Dict:
        """
        Query the summary collection for similar embeddings.
        
        Args:
            query_text: Text query to search for
            n_results: Number of results to return
            filter_criteria: Optional filter criteria
            
        Returns:
            Query results from ChromaDB
        """
        if self.summaries_collection is None:
            raise ValueError("Summary collection not initialized")
        
        logger.info(f"Querying summaries with: '{query_text}'")
        
        results = self.summaries_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_criteria
        )
        
        logger.info(f"Found {len(results['documents'][0])} matching summaries")
        
        return results
    
    def query_all(self, query_text: str, n_results: int = 5) -> Dict[str, Dict]:
        """
        Query both collections and combine results.
        
        Args:
            query_text: Text query to search for
            n_results: Number of results to return from each collection
            
        Returns:
            Dictionary with results from both collections
        """
        transaction_results = self.query_transactions(query_text, n_results)
        summary_results = self.query_summaries(query_text, n_results)
        
        return {
            'transactions': transaction_results,
            'summaries': summary_results
        }
    
    def filter_by_location(self, location: str, collection_type: str = 'both') -> Dict:
        """
        Query embeddings filtered by branch location.
        
        Args:
            location: Branch location to filter by
            collection_type: Type of collection to query ('transactions', 'summaries', or 'both')
            
        Returns:
            Query results matching the location filter
        """
        filter_criteria = {"location": location}
        
        if collection_type == 'transactions' or collection_type == 'both':
            transaction_results = self.query_transactions(
                query_text=f"Transactions at {location}",
                n_results=10,
                filter_criteria=filter_criteria
            )
        else:
            transaction_results = None
        
        if collection_type == 'summaries' or collection_type == 'both':
            summary_results = self.query_summaries(
                query_text=f"Summary for {location}",
                n_results=10,
                filter_criteria=filter_criteria
            )
        else:
            summary_results = None
        
        return {
            'transactions': transaction_results,
            'summaries': summary_results
        }
    
    def filter_by_date_range(self, 
                           start_date: str, 
                           end_date: str, 
                           location: Optional[str] = None) -> Dict:
        """
        Query embeddings filtered by date range and optionally by location.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            location: Optional branch location to filter by
            
        Returns:
            Query results matching the date range and location filter
        """
        # This is a simplified implementation
        # ChromaDB doesn't support date range filtering directly
        # In a real application, you'd need a more sophisticated approach
        
        query_text = f"Transactions between {start_date} and {end_date}"
        if location:
            query_text += f" at {location}"
        
        return self.query_all(query_text)

if __name__ == "__main__":
    # Example usage
    vector_store = TransactionVectorStore(persist_directory="../data/vector_db")
    
    # Load data from embeddings files
    vector_store.load_from_files(
        transaction_embeddings_path="transactions_embeddings.npy",
        transaction_texts_path="transactions_texts.txt",
        transaction_metadata_path="transactions_metadata.csv",
        summary_embeddings_path="summaries_embeddings.npy",
        summary_texts_path="summaries_texts.txt",
        summary_metadata_path="summaries_metadata.csv"
    )
    
    # Example query
    results = vector_store.query_all("Large debit transactions in New York")
    print(f"Query results: {len(results['transactions']['documents'][0])} transactions, "
          f"{len(results['summaries']['documents'][0])} summaries")

