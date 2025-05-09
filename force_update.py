"""
Force Update Script

This script forces the system to process new data, rebuild the vector DB, and generate alerts.
It directly uses the DataVersionManager to ensure the data is properly integrated.
"""

import os
import sys
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("force_update.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def force_update(data_file_path):
    """
    Force the system to update with new data.
    
    Args:
        data_file_path: Path to the new data file
    """
    try:
        if not os.path.exists(data_file_path):
            logger.error(f"Data file not found: {data_file_path}")
            return False
            
        # Import required modules
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from data.data_versioning import DataVersionManager
        from embeddings.vector_store import TransactionVectorStore
        from rag_pipeline.alert_generator import AutomaticAlertGenerator
        
        logger.info(f"Starting forced update with data from: {data_file_path}")
        
        # 1. Process the new data file with merge=False to replace existing data
        version_manager = DataVersionManager(auto_generate_alerts=False)
        initial_version = version_manager.get_current_version()
        logger.info(f"Initial data version: {initial_version}")
        
        # Process the new data file - forcing a new clean version
        logger.info("Processing new data file and creating new version (not merging)")
        new_version = version_manager.process_new_data(
            raw_data_file=data_file_path,
            merge_with_existing=False  # Important: not merging to ensure our data is used
        )
        logger.info(f"Created new data version: {new_version}")
        
        # 2. Wait a moment to ensure file system operations complete
        time.sleep(2)
        
        # 3. Force rebuild of vector store
        logger.info("Forcing vector DB rebuild...")
        result = version_manager.update_vector_db()
        if result.get('status') == 'success':
            logger.info("Vector store updated successfully")
        else:
            logger.error(f"Vector store update failed: {result.get('message')}")
        
        # 4. Wait a moment for vector store to stabilize
        time.sleep(2)
        
        # 5. Force alert generation
        logger.info("Forcing alert generation...")
        alerts = version_manager.generate_alerts()
        logger.info(f"Generated {len(alerts) if alerts else 0} alerts")
        
        logger.info("Forced update completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during forced update: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Use the enhanced test data
    data_file = 'enhanced_test_data.csv'
    
    if not os.path.exists(data_file):
        logger.error(f"Enhanced test data file not found: {data_file}")
        logger.info("Please run enhance_test_data.py first")
        sys.exit(1)
    
    logger.info(f"=== FORCE UPDATE STARTING AT {datetime.now().isoformat()} ===")
    success = force_update(data_file)
    logger.info(f"=== FORCE UPDATE COMPLETED: {'SUCCESS' if success else 'FAILED'} ===")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 