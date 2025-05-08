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
        
        # Initialize version manager
        version_manager = DataVersionManager()
        
        # Get latest data version
        current_version = version_manager.get_current_version()
        logger.info(f"Current data version: {current_version}")
        
        # Update vector DB
        update_info = version_manager.update_vector_db()
        
        if update_info["status"] == "success":
            logger.info(f"Successfully updated vector store with version {current_version}")
            logger.info(f"Update timestamp: {update_info['timestamp']}")
            logger.info(f"Update message: {update_info['message']}")
        else:
            logger.error(f"Failed to update vector store: {update_info['message']}")
            
        return update_info["status"] == "success"
        
    except Exception as e:
        logger.error(f"Error updating vector store: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info(f"Vector store update triggered at {datetime.now().isoformat()}")
    success = update_vectors_after_data_change()
    exit_code = 0 if success else 1
    sys.exit(exit_code) 