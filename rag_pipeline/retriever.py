"""
Retriever Module for the RAG Pipeline.

This module handles retrieving relevant context from the vector database
for generating alerts based on user queries.
"""

import logging
from typing import List, Dict, Any, Optional
import sys
import os
import traceback

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
        if not results:
            return "No relevant information found."
        
        try:
            # Handle potentially missing 'documents' key
            if 'documents' not in results:
                return "No document content available in results."
                
            documents = results.get('documents', [])
            # Handle empty results or first element not being a list
            if not documents or not isinstance(documents[0], list):
                return "No documents found in results."
                
            documents = documents[0]
                
            # Get metadata if available
            metadatas = results.get('metadatas', [])
            if metadatas and isinstance(metadatas[0], list):
                metadatas = metadatas[0]
            else:
                metadatas = None
                
            # Limit the number of items
            documents = documents[:max_items]
            if metadatas:
                metadatas = metadatas[:max_items]
            
            # Format the context
            context = ""
            for i, document in enumerate(documents):
                # Add metadata if available
                if metadatas and i < len(metadatas):
                    metadata = metadatas[i]
                    context += f"--- Item {i+1} ---\n"
                    for key, value in metadata.items():
                        if key != 'id':  # Skip id field
                            context += f"{key}: {value}\n"
                
                # Add document text
                context += f"{document}\n\n"
                
            return context
            
        except Exception as e:
            logger.error(f"Error formatting retrieval results: {e}")
            return f"Error formatting results: {str(e)}"
    
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieve relevant documents based on query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of document texts
        """
        logger.info(f"Retrieving context for query: {query}")
        
        try:
            # Ensure collections are initialized
            if self.vector_store.transactions_collection is None:
                self.vector_store.create_transaction_collection()
            if self.vector_store.summaries_collection is None:
                self.vector_store.create_summary_collection()
                
            # Query transaction collection with proper error handling
            transaction_results = {}
            try:
                transaction_results = self.vector_store.query_transactions(query, n_results=k)
                logger.info(f"Retrieved {len(transaction_results.get('documents', [[]])[0])} transaction results")
            except Exception as tx_error:
                logger.error(f"Error querying transactions: {str(tx_error)}")
                transaction_results = {"documents": [[]], "metadatas": [[]]}
            
            # Query summary collection with proper error handling
            summary_results = {}
            try:
                summary_results = self.vector_store.query_summaries(query, n_results=k)
                logger.info(f"Retrieved {len(summary_results.get('documents', [[]])[0])} summary results")
            except Exception as sum_error:
                logger.error(f"Error querying summaries: {str(sum_error)}")
                summary_results = {"documents": [[]], "metadatas": [[]]}
            
            # Extract documents from both results with careful error handling
            transaction_docs = []
            if 'documents' in transaction_results and transaction_results['documents'] and len(transaction_results['documents']) > 0:
                if isinstance(transaction_results['documents'][0], list):
                    transaction_docs = transaction_results['documents'][0]
                else:
                    transaction_docs = [transaction_results['documents'][0]]
                
            summary_docs = []
            if 'documents' in summary_results and summary_results['documents'] and len(summary_results['documents']) > 0:
                if isinstance(summary_results['documents'][0], list):
                    summary_docs = summary_results['documents'][0]
                else:
                    summary_docs = [summary_results['documents'][0]]
                
            # Combine documents
            combined_docs = transaction_docs + summary_docs
            
            # Filter out empty texts
            filtered_texts = [text for text in combined_docs if text and isinstance(text, str) and text.strip()]
            
            # Limit context length to avoid token issues
            max_chars_per_doc = 2000
            truncated_texts = [text[:max_chars_per_doc] for text in filtered_texts]
            
            if not truncated_texts:
                logger.warning("No relevant documents found for query")
                # Return a placeholder to avoid empty context
                return ["No specific data found for this query."]
                
            return truncated_texts
            
        except Exception as e:
            # Add detailed error info
            error_trace = traceback.format_exc()
            logger.error(f"Error retrieving context: {str(e)}\n{error_trace}")
            return ["Error retrieving context information. Using general knowledge instead."]
    
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
        
        try:
            # Get individual document results
            context_docs = self.retrieve(query, n_results)
            
            # Format results for LLM consumption
            combined_context = "### RETRIEVED CONTEXT:\n\n"
            
            for i, doc in enumerate(context_docs):
                combined_context += f"Document {i+1}:\n{doc}\n\n"
            
            return combined_context
        except Exception as e:
            logger.error(f"Error in retrieve_for_query: {str(e)}")
            return "Error retrieving context. Using general knowledge instead."
    
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

    def augment_query(self, query: str, additional_context: Optional[str] = None) -> str:
        """
        Augment query with retrieved context.
        
        Args:
            query: Original query
            additional_context: Additional context to include
            
        Returns:
            Augmented query
        """
        # Retrieve relevant context
        context_docs = self.retrieve(query)
        
        # Format context
        formatted_context = "\n\n".join([f"Context {i+1}:\n{doc}" for i, doc in enumerate(context_docs)])
        
        # Create augmented query
        augmented_query = f"Query: {query}\n\n"
        
        if context_docs:
            augmented_query += f"Retrieved Context:\n{formatted_context}\n\n"
        
        if additional_context:
            augmented_query += f"Additional Context:\n{additional_context}\n\n"
        
        return augmented_query

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

