"""
Vector Store Update Script

This script updates the vector store when data changes.
It is automatically triggered by DVC hooks when data is pulled or updated.
"""

import os
import sys
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vector_store_updates.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def update_vectors_after_data_change():
    """Update the vector store based on the latest data version."""
    try:
        # Import the DataVersionManager after adding project root to sys.path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from data.data_versioning import DataVersionManager
        
        logger.info("Starting vector store update process")
        
        # Initialize the data version manager with auto-alert generation
        version_manager = DataVersionManager(auto_generate_alerts=True)
        
        # Update the vector database
        result = version_manager.update_vector_db()
        
        if result.get('status') == 'success':
            logger.info("Vector store update completed successfully")
            logger.info("Automatic alert generation has been triggered")
        else:
            logger.error(f"Vector store update failed: {result.get('message')}")
            
        return result.get('status') == 'success'
    except Exception as e:
        logger.error(f"Error updating vector store: {e}")
        return False

if __name__ == "__main__":
    logger.info(f"Vector store update triggered at {datetime.now().isoformat()}")
    success = update_vectors_after_data_change()
    exit_code = 0 if success else 1
    sys.exit(exit_code) 