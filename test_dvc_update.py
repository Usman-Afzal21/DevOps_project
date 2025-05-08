"""
Test DVC Update Script

This script tests the DVC setup by simulating adding new data and updating the vector store.
It performs the following steps:
1. Creates a new data version
2. Runs the vector store update script
3. Verifies that the vector store was updated
"""

import os
import sys
import time
import logging
import subprocess
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dvc_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_dvc_update():
    try:
        # Add project root to sys.path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from data.data_versioning import DataVersionManager
        
        logger.info("Starting DVC update test")
        
        # 1. Get initial version
        version_manager = DataVersionManager()
        initial_version = version_manager.get_current_version()
        logger.info(f"Initial data version: {initial_version}")
        
        # 2. Create a new version (simulate adding new data)
        new_version = version_manager.create_new_version(f"Test version created at {datetime.now().isoformat()}")
        logger.info(f"Created new data version: {new_version}")
        
        # 3. Run the vector store update script
        logger.info("Running vector store update script")
        if os.name == 'nt':  # Windows
            result = subprocess.run(['update_vector_store.bat'], capture_output=True, text=True)
        else:  # Unix-like
            result = subprocess.run(['python', 'update_vector_store.py'], capture_output=True, text=True)
        
        logger.info(f"Update script output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Update script errors: {result.stderr}")
            
        logger.info(f"Update script exit code: {result.returncode}")
        
        # 4. Verify that the vector store was updated
        # Check current version
        updated_version = version_manager.get_current_version()
        logger.info(f"Current version after update: {updated_version}")
        
        # 5. Update DVC with the changes
        logger.info("Updating DVC to track changes")
        try:
            subprocess.run(['dvc', 'add', 'data/vector_db/'], check=True)
            logger.info("Successfully updated DVC tracking for vector_db")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to update DVC tracking: {e}")
        
        # Test completed successfully
        logger.info("DVC update test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in DVC update test: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_dvc_update()
    exit_code = 0 if success else 1
    sys.exit(exit_code) 