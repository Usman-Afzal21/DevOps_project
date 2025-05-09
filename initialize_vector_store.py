"""
Initialize Vector Store Script

This script initializes the vector store with the base data in the data/processed folder.
This is a one-time operation to create the vector embeddings for RAG functionality.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vector_store_initialization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def initialize_base_vector_store():
    """Initialize the vector store with only the base data in the data/processed folder."""
    try:
        # Import necessary modules
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from embeddings.embedder import TransactionEmbedder
        from embeddings.vector_store import TransactionVectorStore
        from data.processed.data_processor import TransactionDataProcessor
        
        logger.info("Starting vector store initialization with base data only")
        
        # Initialize the vector store
        vector_store = TransactionVectorStore(persist_directory="data/vector_db")
        
        # Create collections for transactions and summaries
        try:
            vector_store.create_transaction_collection()
            vector_store.create_summary_collection()
            logger.info("Created vector store collections")
        except Exception as e:
            logger.error(f"Error creating collections: {e}")
            return {"status": "error", "message": str(e)}
        
        # Initialize the embedder
        embedder = TransactionEmbedder()
        
        # Base data paths
        processed_data_dir = os.path.join("data", "processed")
        transactions_path = os.path.join(processed_data_dir, "processed_transactions.csv")
        summaries_path = os.path.join(processed_data_dir, "branch_summaries.csv")
        
        # Check if necessary files exist
        if not os.path.exists(transactions_path) or not os.path.exists(summaries_path):
            logger.warning("Base data files not found. Processing raw data first.")
            
            # Initialize the processor and process the data
            processor = TransactionDataProcessor(
                raw_data_path=os.path.join("data", "raw"),
                processed_data_path=processed_data_dir
            )
            
            try:
                processor.process_all_data_pipeline()
                logger.info("Successfully processed raw data")
            except Exception as e:
                logger.error(f"Error processing raw data: {e}")
                return {"status": "error", "message": f"Error processing raw data: {str(e)}"}
        
        # Load transaction data
        try:
            transactions_df = pd.read_csv(transactions_path)
            logger.info(f"Loaded {len(transactions_df)} transaction records")
        except Exception as e:
            logger.error(f"Error loading transaction data: {e}")
            return {"status": "error", "message": str(e)}
        
        # Load summary data
        try:
            summaries_df = pd.read_csv(summaries_path)
            logger.info(f"Loaded {len(summaries_df)} branch summary records")
        except Exception as e:
            logger.error(f"Error loading summary data: {e}")
            return {"status": "error", "message": str(e)}
        
        # Generate transaction embeddings
        try:
            txn_embeddings = embedder.generate_transaction_embeddings(transactions_df)
            logger.info(f"Generated embeddings for {len(transactions_df)} transactions")
            
            # Add transaction embeddings to vector store
            vector_store.add_transaction_embeddings(
                embeddings=txn_embeddings["embeddings"],
                texts=txn_embeddings["texts"],
                ids=txn_embeddings["ids"],
                metadata=txn_embeddings["metadata"]
            )
            logger.info("Added transaction embeddings to vector store")
        except Exception as e:
            logger.error(f"Error generating transaction embeddings: {e}")
            return {"status": "error", "message": str(e)}
        
        # Generate summary embeddings
        try:
            summary_embeddings = embedder.generate_summary_embeddings(summaries_df)
            logger.info(f"Generated embeddings for {len(summaries_df)} branch summaries")
            
            # Add summary embeddings to vector store
            vector_store.add_summary_embeddings(
                embeddings=summary_embeddings["embeddings"],
                texts=summary_embeddings["texts"],
                ids=summary_embeddings["ids"],
                metadata=summary_embeddings["metadata"]
            )
            logger.info("Added summary embeddings to vector store")
        except Exception as e:
            logger.error(f"Error generating summary embeddings: {e}")
            return {"status": "error", "message": str(e)}
        
        logger.info("Vector store initialization completed successfully")
        return {"status": "success", "message": "Vector store initialized with base data"}
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    logger.info(f"Vector store initialization started at {datetime.now().isoformat()}")
    result = initialize_base_vector_store()
    
    if result["status"] == "success":
        logger.info("Vector store initialized successfully")
        logger.info("Base data is now available for RAG functionality")
        logger.info("Note: This script only needs to be run once or when base data changes")
        sys.exit(0)
    else:
        logger.error(f"Vector store initialization failed: {result['message']}")
        sys.exit(1) 