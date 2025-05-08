"""
Add New Data Script

This script simplifies the process of adding new data and updating the vector store.
Usage: python add_new_data.py path/to/new_data_file.csv [--no-merge]
"""

import os
import sys
import argparse
import logging
import subprocess
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_updates.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def add_new_data(data_file_path, merge_with_existing=True):
    """
    Add new data and update the vector store.
    
    Args:
        data_file_path: Path to the new data file
        merge_with_existing: Whether to merge with existing data
    
    Returns:
        Success flag
    """
    try:
        if not os.path.exists(data_file_path):
            logger.error(f"Data file not found: {data_file_path}")
            return False
            
        # Import the DataVersionManager
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from data.data_versioning import DataVersionManager
        
        logger.info(f"Adding new data from: {data_file_path}")
        logger.info(f"Merge with existing: {merge_with_existing}")
        
        # Process new data
        version_manager = DataVersionManager()
        initial_version = version_manager.get_current_version()
        logger.info(f"Initial data version: {initial_version}")
        
        # Process the new data file
        new_version = version_manager.process_new_data(
            raw_data_file=data_file_path,
            merge_with_existing=merge_with_existing
        )
        logger.info(f"Created new data version: {new_version}")
        
        # Update DVC tracking
        logger.info("Updating DVC tracking")
        try:
            subprocess.run(['dvc', 'add', 'data/raw/', 'data/processed/', 'data/vector_db/'], check=True)
            logger.info("Successfully updated DVC tracking")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to update DVC tracking: {e}")
            
        # Auto-commit if git is available and initialized
        try:
            # Check if inside a git repo
            git_check = subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'], 
                                     capture_output=True, text=True)
            
            if git_check.returncode == 0 and git_check.stdout.strip() == 'true':
                logger.info("Committing changes to git")
                commit_msg = f"Add new data and update vector store (version {new_version})"
                
                # Add changes to git
                subprocess.run(['git', 'add', '.gitignore', '*.dvc'], check=True)
                subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
                logger.info("Successfully committed changes to git")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not auto-commit changes: {e}")
        except FileNotFoundError:
            logger.warning("Git not found, skipping auto-commit")
            
        logger.info(f"Successfully added new data and updated vector store. New version: {new_version}")
        return True
        
    except Exception as e:
        logger.error(f"Error adding new data: {str(e)}")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Add new data and update vector store")
    parser.add_argument("data_file", help="Path to the new data file")
    parser.add_argument("--no-merge", action="store_true", help="Do not merge with existing data")
    
    args = parser.parse_args()
    
    # Run the data addition process
    success = add_new_data(args.data_file, not args.no_merge)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 