"""
Retriever Module for the RAG Pipeline.

This module handles retrieving relevant context from the vector database
for generating alerts based on user queries.
"""

import logging
from typing import List, Dict, Any, Optional
import sys
import os

# Add embeddings directory to path to import TransactionVectorStore
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from embeddings.vector_store import TransactionVectorStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TransactionRetriever:
    """
    Retrieve relevant transaction context for RAG alerts.
    """
    
    def __init__(self, vector_store: TransactionVectorStore):
        """
        Initialize the retriever with a vector store.
        
        Args:
            vector_store: An initialized TransactionVectorStore
        """
        self.vector_store = vector_store
    
    def format_retrieval_results(self, results: Dict[str, Any], max_items: int = 5) -> str:
        """
        Format retrieval results into a string for context.
        
        Args:
            results: Results from vector store query
            max_items: Maximum number of items to include
            
        Returns:
            Formatted context string
        """
        if not results or 'documents' not in results or not results['documents']:
            return "No relevant information found."
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if 'metadatas' in results else None
        
        # Limit the number of items
        documents = documents[:max_items]
        if metadatas:
            metadatas = metadatas[:max_items]
        
        # Format the context
        context = ""
        for i, document in enumerate(documents):
            # Add metadata if available
            if metadatas:
                metadata = metadatas[i]
                context += f"--- Item {i+1} ---\n"
                for key, value in metadata.items():
                    if key != 'id':  # Skip id field
                        context += f"{key}: {value}\n"
            
            # Add document text
            context += f"{document}\n\n"
        
        return context
    
    def retrieve_for_query(self, query: str, n_results: int = 5) -> str:
        """
        Retrieve context based on a user query.
        
        Args:
            query: The user's query about transaction data
            n_results: Number of results to retrieve
            
        Returns:
            Formatted context string
        """
        logger.info(f"Retrieving context for query: {query}")
        
        # Query both collections
        all_results = self.vector_store.query_all(query, n_results)
        
        # Format results
        transaction_context = self.format_retrieval_results(
            all_results['transactions'], max_items=n_results
        )
        summary_context = self.format_retrieval_results(
            all_results['summaries'], max_items=n_results
        )
        
        # Combine contexts
        combined_context = "TRANSACTION DETAILS:\n" + transaction_context + "\n\n"
        combined_context += "BRANCH SUMMARIES:\n" + summary_context
        
        return combined_context
    
    def retrieve_for_branch(self, branch: str, n_results: int = 5) -> str:
        """
        Retrieve context for a specific branch.
        
        Args:
            branch: The branch to retrieve context for
            n_results: Number of results to retrieve
            
        Returns:
            Formatted context string
        """
        logger.info(f"Retrieving context for branch: {branch}")
        
        # Get branch-specific results
        branch_results = self.vector_store.filter_by_location(branch)
        
        # Format results
        transaction_context = self.format_retrieval_results(
            branch_results['transactions'], max_items=n_results
        )
        summary_context = self.format_retrieval_results(
            branch_results['summaries'], max_items=n_results
        )
        
        # Combine contexts
        combined_context = f"TRANSACTION DETAILS FOR {branch}:\n" + transaction_context + "\n\n"
        combined_context += f"BRANCH SUMMARIES FOR {branch}:\n" + summary_context
        
        return combined_context
    
    def retrieve_for_date_range(self, start_date: str, end_date: str, 
                              branch: Optional[str] = None, n_results: int = 5) -> str:
        """
        Retrieve context for a specific date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            branch: Optional branch to filter by
            n_results: Number of results to retrieve
            
        Returns:
            Formatted context string
        """
        logger.info(f"Retrieving context for date range: {start_date} to {end_date}")
        
        # Get date range results
        date_results = self.vector_store.filter_by_date_range(start_date, end_date, branch)
        
        # Format results
        transaction_context = self.format_retrieval_results(
            date_results['transactions'], max_items=n_results
        )
        summary_context = self.format_retrieval_results(
            date_results['summaries'], max_items=n_results
        )
        
        # Combine contexts
        time_period = f"from {start_date} to {end_date}"
        branch_str = f" for {branch}" if branch else ""
        
        combined_context = f"TRANSACTION DETAILS {time_period}{branch_str}:\n" + transaction_context + "\n\n"
        combined_context += f"BRANCH SUMMARIES {time_period}{branch_str}:\n" + summary_context
        
        return combined_context
    
    def retrieve_for_comparison(self, branch_a: str, branch_b: str, n_results: int = 5) -> Dict[str, str]:
        """
        Retrieve context for comparing two branches.
        
        Args:
            branch_a: First branch to compare
            branch_b: Second branch to compare
            n_results: Number of results to retrieve for each branch
            
        Returns:
            Dictionary with context for each branch
        """
        logger.info(f"Retrieving context for branch comparison: {branch_a} vs {branch_b}")
        
        # Get branch-specific results
        branch_a_results = self.vector_store.filter_by_location(branch_a)
        branch_b_results = self.vector_store.filter_by_location(branch_b)
        
        # Format results
        branch_a_context = self.format_retrieval_results(
            branch_a_results['summaries'], max_items=n_results
        )
        branch_b_context = self.format_retrieval_results(
            branch_b_results['summaries'], max_items=n_results
        )
        
        return {
            'branch_a_context': branch_a_context,
            'branch_b_context': branch_b_context
        }

if __name__ == "__main__":
    # Example usage
    vector_store = TransactionVectorStore(persist_directory="../data/vector_db")
    
    # Load data (assuming vector store has been created)
    try:
        vector_store.create_transaction_collection()
        vector_store.create_summary_collection()
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        
    retriever = TransactionRetriever(vector_store)
    
    # Example retrieval
    context = retriever.retrieve_for_query("large debit transactions with high fraud risk")
    print(f"Retrieved context length: {len(context)} characters")

