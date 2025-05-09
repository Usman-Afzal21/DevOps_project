"""
RAG System Setup Script

This script initializes the banking RAG system by:
1. Processing the base data in data/raw
2. Initializing the vector store once with the base data
3. Setting up the system for more efficient direct data analysis

After running this script, the system will be ready to accept new data uploads
through the UI without needing to re-vectorize the entire dataset each time.
"""

import os
import sys
import logging
import subprocess
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("setup_rag_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_rag_system():
    """
    Set up the RAG system with base data for more efficient operation.
    """
    logger.info("Starting RAG system setup")
    
    # Step 1: Process base data in data/raw
    logger.info("Step 1: Processing base data")
    try:
        # Import the DataVersionManager
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from data.processed.data_processor import TransactionDataProcessor
        
        # Initialize processor for base data
        processor = TransactionDataProcessor(
            raw_data_path="data/raw",
            processed_data_path="data/processed"
        )
        
        # Process all raw data
        logger.info("Processing all raw data files")
        output_files = processor.process_all_data_pipeline()
        logger.info(f"Processed data files: {output_files}")
    except Exception as e:
        logger.error(f"Error processing base data: {e}")
        return False
    
    # Step 2: Initialize vector store with base data
    logger.info("Step 2: Initializing vector store with base data")
    try:
        logger.info("Running vector store initialization script")
        result = subprocess.run([sys.executable, "initialize_vector_store.py"], 
                               capture_output=True, 
                               text=True)
        
        if result.returncode != 0:
            logger.error(f"Vector store initialization failed: {result.stderr}")
            return False
        
        logger.info("Vector store initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        return False
    
    # Step 3: Create necessary directories if they don't exist
    logger.info("Step 3: Creating necessary directories")
    try:
        os.makedirs("alerts", exist_ok=True)
        os.makedirs("data/versions", exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
    
    logger.info("RAG system setup completed successfully")
    return True

def display_post_setup_instructions():
    """
    Display instructions for using the system after setup.
    """
    print("\n" + "="*80)
    print("RAG SYSTEM SETUP COMPLETE")
    print("="*80)
    print("\nYour banking RAG system has been set up for efficient operation.")
    print("\nWhat's been done:")
    print("1. Base data in data/raw has been processed")
    print("2. Vector store has been initialized with base data")
    print("3. System configured for direct analysis of new data")
    print("\nHow to use the system:")
    print("1. Run the Streamlit app:   streamlit run app.py")
    print("2. Start the API server:    python api/main.py")
    print("3. Upload new CSV data through the UI")
    print("\nThe system will now:")
    print("- Analyze new data directly without re-vectorizing everything")
    print("- Generate alerts based on anomalies in new data")
    print("- Use RAG to provide context from the base data")
    print("- Only version the new data with DVC")
    print("\nThis is much more efficient for large datasets!")
    print("="*80)

if __name__ == "__main__":
    logger.info(f"RAG system setup started at {datetime.now().isoformat()}")
    
    if setup_rag_system():
        logger.info("RAG system setup completed successfully")
        display_post_setup_instructions()
        sys.exit(0)
    else:
        logger.error("RAG system setup failed")
        print("\nERROR: RAG system setup failed. Check setup_rag_system.log for details.")
        sys.exit(1) 