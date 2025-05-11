"""
Data Versioning Module for MLOps Pipeline.

This module handles:
- Data version control
- Dataset updating with new data
- Tracking data lineage
- Integration with the RAG pipeline
- Automatic alert generation
"""

import os
import json
import shutil
import logging
import datetime
import pandas as pd
from typing import Dict, Any, List, Optional
import sys

# Add project directories to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.processed.data_processor import TransactionDataProcessor
from embeddings.embedder import TransactionEmbedder
from embeddings.vector_store import TransactionVectorStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataVersionManager:
    """
    Manage data versions, updates, and lineage for the MLOps pipeline.
    """
    
    def __init__(
        self, 
        base_dir: str = os.path.join(os.path.dirname(__file__), '..'),
        version_dir: str = "data/versions",
        metadata_file: str = "data/version_metadata.json",
        auto_generate_alerts: bool = True
    ):
        """
        Initialize the data version manager.
        
        Args:
            base_dir: Base directory of the project
            version_dir: Directory to store versioned data
            metadata_file: File to store version metadata
            auto_generate_alerts: Whether to auto-generate alerts on data update
        """
        self.base_dir = base_dir
        self.version_dir = os.path.join(base_dir, version_dir)
        self.metadata_file = os.path.join(base_dir, metadata_file)
        self.raw_data_dir = os.path.join(base_dir, "data/raw")
        self.processed_data_dir = os.path.join(base_dir, "data/processed")
        self.embeddings_dir = os.path.join(base_dir, "embeddings")
        self.vector_db_dir = os.path.join(base_dir, "data/vector_db")
        self.auto_generate_alerts = auto_generate_alerts
        
        # Create necessary directories
        os.makedirs(self.version_dir, exist_ok=True)
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.vector_db_dir, exist_ok=True)
        
        # Initialize or load version metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load version metadata from file or initialize if it doesn't exist.
        
        Returns:
            Dictionary of version metadata
        """
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata file: {e}")
                return self._initialize_metadata()
        else:
            return self._initialize_metadata()
    
    def _initialize_metadata(self) -> Dict[str, Any]:
        """
        Initialize version metadata structure.
        
        Returns:
            Dictionary of initialized version metadata
        """
        metadata = {
            "current_version": "v0.0.0",
            "versions": {},
            "data_lineage": [],
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        self._save_metadata(metadata)
        return metadata
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """
        Save version metadata to file.
        
        Args:
            metadata: Dictionary of version metadata
        """
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_current_version(self) -> str:
        """
        Get the current data version.
        
        Returns:
            Current version string
        """
        return self.metadata["current_version"]
    
    def get_version_info(self, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a specific version.
        
        Args:
            version: Version to get info for. If None, uses current version.
            
        Returns:
            Dictionary of version information
        """
        if version is None:
            version = self.get_current_version()
            
        if version in self.metadata["versions"]:
            return self.metadata["versions"][version]
        else:
            raise ValueError(f"Version {version} not found in metadata")
    
    def list_versions(self) -> List[str]:
        """
        List all available data versions.
        
        Returns:
            List of version strings
        """
        return list(self.metadata["versions"].keys())
    
    def create_new_version(self, version_notes: str = "") -> str:
        """
        Create a new data version.
        
        Args:
            version_notes: Notes about this version
            
        Returns:
            New version string
        """
        # Parse current version
        current = self.metadata["current_version"]
        if current == "v0.0.0":
            new_version = "v1.0.0"
        else:
            try:
                # Remove 'v' prefix and split
                major, minor, patch = map(int, current[1:].split('.'))
                # Increment major version
                new_version = f"v{major + 1}.0.0"
            except Exception:
                # Fallback to timestamp if version parsing fails
                new_version = f"v{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create version directory
        version_dir = os.path.join(self.version_dir, new_version)
        os.makedirs(version_dir, exist_ok=True)
        os.makedirs(os.path.join(version_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(version_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(version_dir, "embeddings"), exist_ok=True)
        os.makedirs(os.path.join(version_dir, "vector_db"), exist_ok=True)
        
        # Create version metadata
        version_metadata = {
            "created_at": datetime.datetime.now().isoformat(),
            "notes": version_notes,
            "parent_version": current,
            "files": {
                "raw": [],
                "processed": [],
                "embeddings": [],
                "vector_db": []
            },
            "metrics": {},
            "processing_info": {}
        }
        
        # Update metadata
        self.metadata["versions"][new_version] = version_metadata
        self.metadata["current_version"] = new_version
        self.metadata["last_updated"] = datetime.datetime.now().isoformat()
        self.metadata["data_lineage"].append({
            "version": new_version,
            "parent": current,
            "timestamp": datetime.datetime.now().isoformat(),
            "operation": "create_new_version",
            "notes": version_notes
        })
        
        self._save_metadata(self.metadata)
        logger.info(f"Created new data version: {new_version}")
        
        return new_version
    
    def process_new_data(self, raw_data_file: str, merge_with_existing: bool = True) -> str:
        """
        Process new data file and create a new version.
        UPDATED: No longer re-vectorizes all data, only versions the new data.
        
        Args:
            raw_data_file: Path to new raw data file
            merge_with_existing: Whether to merge with existing data
            
        Returns:
            New version string
        """
        logger.info(f"Processing new data file: {raw_data_file}")
        
        # Create new version
        new_version = self.create_new_version(f"Added new data from {os.path.basename(raw_data_file)}")
        version_dir = os.path.join(self.version_dir, new_version)
        
        # Copy new raw data to version directory and current raw directory
        new_raw_path = os.path.join(version_dir, "raw", os.path.basename(raw_data_file))
        shutil.copy2(raw_data_file, new_raw_path)
        current_raw_path = os.path.join(self.raw_data_dir, os.path.basename(raw_data_file))
        shutil.copy2(raw_data_file, current_raw_path)
        
        # Update metadata for raw files
        self.metadata["versions"][new_version]["files"]["raw"].append(os.path.basename(raw_data_file))
        
        # Process new data
        if merge_with_existing:
            # If merging with existing data, load both datasets
            current_data = None
            try:
                current_processed_path = os.path.join(self.processed_data_dir, "processed_transactions.csv")
                if os.path.exists(current_processed_path):
                    current_data = pd.read_csv(current_processed_path)
                    logger.info(f"Loaded existing data with {len(current_data)} records")
            except Exception as e:
                logger.warning(f"Error loading existing data: {e}. Will process new data only.")
            
            # Load new data
            try:
                new_data = pd.read_csv(raw_data_file)
                logger.info(f"Loaded new data with {len(new_data)} records")
                
                # Merge datasets if current data exists
                if current_data is not None:
                    merged_data = pd.concat([current_data, new_data], ignore_index=True)
                    logger.info(f"Merged data has {len(merged_data)} records")
                else:
                    merged_data = new_data
                    logger.info(f"No existing data found. Using only new data ({len(merged_data)} records)")
                
                # Initialize data processor
                processor = TransactionDataProcessor(
                    raw_data_path=self.raw_data_dir,
                    processed_data_path=self.processed_data_dir
                )
                
                # Process the merged data
                processor.data = merged_data
                processor.clean_data()
                processor.create_features()
                
                # Save processed data
                saved_files = processor.save_processed_data()
                
                # Copy processed files to version directory
                for file_type, file_path in saved_files.items():
                    filename = os.path.basename(file_path)
                    version_processed_path = os.path.join(version_dir, "processed", filename)
                    shutil.copy2(file_path, version_processed_path)
                    self.metadata["versions"][new_version]["files"]["processed"].append(filename)
                
                # Update processing info
                self.metadata["versions"][new_version]["processing_info"] = {
                    "processed_records": len(merged_data),
                    "processed_files": list(saved_files.values()),
                    "merge_with_existing": merge_with_existing
                }
                
                # Save the metadata
                self._save_metadata(self.metadata)
                
                # Directly analyze the new data without updating vectors
                self._analyze_new_data(new_data)
                
                return new_version
                
            except Exception as e:
                logger.error(f"Error processing new data: {e}")
                raise
        else:
            # Process new data only
            try:
                new_data = pd.read_csv(raw_data_file)
                logger.info(f"Loaded new data with {len(new_data)} records")
                
                # Initialize data processor
                processor = TransactionDataProcessor(
                    raw_data_path=self.raw_data_dir,
                    processed_data_path=self.processed_data_dir
                )
                
                # Process the new data
                processor.data = new_data
                processor.clean_data()
                processor.create_features()
                
                # Save processed data
                saved_files = processor.save_processed_data()
                
                # Copy processed files to version directory
                for file_type, file_path in saved_files.items():
                    filename = os.path.basename(file_path)
                    version_processed_path = os.path.join(version_dir, "processed", filename)
                    shutil.copy2(file_path, version_processed_path)
                    self.metadata["versions"][new_version]["files"]["processed"].append(filename)
                
                # Update processing info
                self.metadata["versions"][new_version]["processing_info"] = {
                    "processed_records": len(new_data),
                    "processed_files": list(saved_files.values()),
                    "merge_with_existing": merge_with_existing
                }
                
                # Save the metadata
                self._save_metadata(self.metadata)
                
                # Directly analyze the new data without updating vectors
                self._analyze_new_data(new_data)
                
                return new_version
                
            except Exception as e:
                logger.error(f"Error processing new data: {e}")
                raise
    
    def _analyze_new_data(self, new_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze new data directly using the RAG pipeline without re-vectorizing.
        
        Args:
            new_data: DataFrame with new transaction data
            
        Returns:
            List of generated alerts
        """
        try:
            # Import analyzer here to avoid circular imports
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from rag_pipeline.direct_data_analyzer import DirectDataAnalyzer
            
            logger.info("Analyzing new data directly with RAG")
            
            # Initialize analyzer
            analyzer = DirectDataAnalyzer()
            
            # Always analyze the data, even if there are no obvious anomalies
            logger.info("Forcing analysis of uploaded data for dashboard insights")
            
            # Make sure data has required columns for analysis
            if 'TransactionID' not in new_data.columns:
                # Add a TransactionID column if it doesn't exist
                new_data['TransactionID'] = [f"TX{str(i+1000).zfill(6)}" for i in range(len(new_data))]
                logger.info("Added TransactionID column to data for analysis")
            
            # Add FraudFlag column if it doesn't exist (for anomaly detection)
            if 'FraudFlag' not in new_data.columns:
                # Check if there's a FraudRiskScore column that could be used
                if 'FraudRiskScore' in new_data.columns:
                    # Convert high risk scores to fraud flags
                    new_data['FraudFlag'] = new_data['FraudRiskScore'] > 0.7
                else:
                    # Set default to False
                    new_data['FraudFlag'] = False
                logger.info("Added FraudFlag column to data for analysis")
            
            # Analyze the data and generate alerts
            alerts = analyzer.analyze_new_data(new_data)
            
            logger.info(f"Generated {len(alerts)} alerts from new data")
            return alerts
        except Exception as e:
            logger.error(f"Error analyzing new data: {e}")
            return []
    
    def update_vector_db(self) -> Dict[str, Any]:
        """
        Update the vector database with the latest embeddings.
        
        Returns:
            Dictionary with update information
        """
        logger.info("Updating vector database with latest embeddings")
        
        # Initialize vector store
        vector_store = TransactionVectorStore(persist_directory=self.vector_db_dir)
        
        # Create collections
        vector_store.create_transaction_collection()
        vector_store.create_summary_collection()
        
        # Load embeddings
        try:
            vector_store.load_from_files(
                transaction_embeddings_path=os.path.join(self.embeddings_dir, "transactions_embeddings.npy"),
                transaction_texts_path=os.path.join(self.embeddings_dir, "transactions_texts.txt"),
                transaction_metadata_path=os.path.join(self.embeddings_dir, "transactions_metadata.csv"),
                summary_embeddings_path=os.path.join(self.embeddings_dir, "summaries_embeddings.npy"),
                summary_texts_path=os.path.join(self.embeddings_dir, "summaries_texts.txt"),
                summary_metadata_path=os.path.join(self.embeddings_dir, "summaries_metadata.csv")
            )
            
            # Create backup of vector db in current version directory
            current_version = self.get_current_version()
            if current_version in self.metadata["versions"]:
                version_vector_db = os.path.join(self.version_dir, current_version, "vector_db")
                self._backup_directory(self.vector_db_dir, version_vector_db)
                
                # Update metadata for vector db files
                db_files = [f for f in os.listdir(self.vector_db_dir) 
                           if os.path.isfile(os.path.join(self.vector_db_dir, f))]
                self.metadata["versions"][current_version]["files"]["vector_db"] = db_files
            
            update_info = {
                "status": "success",
                "timestamp": datetime.datetime.now().isoformat(),
                "message": "Vector database updated successfully"
            }
            
            logger.info("Vector database updated successfully")
            
            # Automatically generate alerts after vector DB update if enabled
            if self.auto_generate_alerts:
                self.generate_alerts()
            
        except Exception as e:
            logger.error(f"Error updating vector database: {e}")
            update_info = {
                "status": "error",
                "timestamp": datetime.datetime.now().isoformat(),
                "message": f"Error updating vector database: {str(e)}"
            }
        
        return update_info
    
    def _backup_directory(self, source_dir: str, target_dir: str):
        """
        Create a backup of a directory.
        
        Args:
            source_dir: Source directory to backup
            target_dir: Target directory for backup
        """
        if os.path.exists(source_dir):
            # Create target directory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy all files
            for item in os.listdir(source_dir):
                s = os.path.join(source_dir, item)
                d = os.path.join(target_dir, item)
                if os.path.isfile(s):
                    shutil.copy2(s, d)
                elif os.path.isdir(s):
                    self._backup_directory(s, d)
    
    def generate_alerts(self) -> List[Dict[str, Any]]:
        """
        Generate alerts based on the latest data.
        
        Returns:
            List of generated alerts
        """
        logger.info("Generating automatic alerts based on latest data")
        
        try:
            # Import the AutomaticAlertGenerator
            from rag_pipeline.alert_generator import AutomaticAlertGenerator
            
            # Initialize alert generator
            alert_generator = AutomaticAlertGenerator(
                processed_data_dir=self.processed_data_dir,
                vector_db_dir=self.vector_db_dir
            )
            
            # Run alert generation process
            alerts = alert_generator.run()
            
            logger.info(f"Generated {len(alerts)} alerts automatically")
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating automatic alerts: {e}")
            return []
    
    def rollback_to_version(self, version: str) -> bool:
        """
        Rollback to a previous data version.
        
        Args:
            version: Version to rollback to
            
        Returns:
            Success flag
        """
        if version not in self.metadata["versions"]:
            logger.error(f"Version {version} not found")
            return False
        
        logger.info(f"Rolling back to version: {version}")
        
        try:
            # Create restore point of current state
            current = self.get_current_version()
            restore_version = self.create_new_version(f"Restore point before rollback to {version}")
            
            # Copy files from version to current directories
            version_dir = os.path.join(self.version_dir, version)
            
            # Restore processed data
            processed_version_dir = os.path.join(version_dir, "processed")
            if os.path.exists(processed_version_dir):
                for file in os.listdir(processed_version_dir):
                    source = os.path.join(processed_version_dir, file)
                    target = os.path.join(self.processed_data_dir, file)
                    if os.path.isfile(source):
                        shutil.copy2(source, target)
            
            # Restore embeddings
            embeddings_version_dir = os.path.join(version_dir, "embeddings")
            if os.path.exists(embeddings_version_dir):
                for file in os.listdir(embeddings_version_dir):
                    source = os.path.join(embeddings_version_dir, file)
                    target = os.path.join(self.embeddings_dir, file)
                    if os.path.isfile(source):
                        shutil.copy2(source, target)
            
            # Restore vector db
            vector_db_version_dir = os.path.join(version_dir, "vector_db")
            if os.path.exists(vector_db_version_dir):
                # Clear current vector db
                if os.path.exists(self.vector_db_dir):
                    shutil.rmtree(self.vector_db_dir)
                    os.makedirs(self.vector_db_dir)
                
                # Copy version vector db
                self._backup_directory(vector_db_version_dir, self.vector_db_dir)
            
            # Update metadata
            self.metadata["current_version"] = version
            self.metadata["last_updated"] = datetime.datetime.now().isoformat()
            self.metadata["data_lineage"].append({
                "version": version,
                "parent": current,
                "timestamp": datetime.datetime.now().isoformat(),
                "operation": "rollback_to_version",
                "notes": f"Rolled back from {current} to {version}"
            })
            
            self._save_metadata(self.metadata)
            logger.info(f"Successfully rolled back to version {version}")
            
            # Generate alerts after rollback if enabled
            if self.auto_generate_alerts:
                self.generate_alerts()
            
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back to version {version}: {e}")
            return False
    
    def compare_versions(self, version_a: str, version_b: str) -> Dict[str, Any]:
        """
        Compare two data versions.
        
        Args:
            version_a: First version
            version_b: Second version
            
        Returns:
            Dictionary with comparison information
        """
        if version_a not in self.metadata["versions"]:
            raise ValueError(f"Version {version_a} not found")
        if version_b not in self.metadata["versions"]:
            raise ValueError(f"Version {version_b} not found")
        
        # Get version info
        info_a = self.metadata["versions"][version_a]
        info_b = self.metadata["versions"][version_b]
        
        # Check processed data files
        try:
            # Load transaction data
            trans_a_path = os.path.join(self.version_dir, version_a, "processed", "processed_transactions.csv")
            trans_b_path = os.path.join(self.version_dir, version_b, "processed", "processed_transactions.csv")
            
            if os.path.exists(trans_a_path) and os.path.exists(trans_b_path):
                trans_a = pd.read_csv(trans_a_path)
                trans_b = pd.read_csv(trans_b_path)
                
                # Compare record counts
                trans_diff = len(trans_b) - len(trans_a)
                
                # Find new transaction IDs
                if 'TransactionID' in trans_a.columns and 'TransactionID' in trans_b.columns:
                    ids_a = set(trans_a['TransactionID'])
                    ids_b = set(trans_b['TransactionID'])
                    new_ids = ids_b - ids_a
                    n_new_records = len(new_ids)
                else:
                    new_ids = set()
                    n_new_records = 0
            else:
                trans_diff = 0
                n_new_records = 0
                new_ids = set()
        except Exception as e:
            logger.error(f"Error comparing transaction data: {e}")
            trans_diff = "Error"
            n_new_records = "Error"
            new_ids = set()
        
        # Compare creation dates
        created_a = datetime.datetime.fromisoformat(info_a["created_at"]) if "created_at" in info_a else None
        created_b = datetime.datetime.fromisoformat(info_b["created_at"]) if "created_at" in info_b else None
        
        if created_a and created_b:
            time_diff = created_b - created_a
            time_diff_days = time_diff.total_seconds() / (60 * 60 * 24)
        else:
            time_diff_days = None
        
        # Build comparison
        comparison = {
            "version_a": version_a,
            "version_b": version_b,
            "creation_date_a": info_a.get("created_at", "Unknown"),
            "creation_date_b": info_b.get("created_at", "Unknown"),
            "time_difference_days": time_diff_days,
            "transaction_count_difference": trans_diff,
            "new_records_count": n_new_records,
            "notes_a": info_a.get("notes", ""),
            "notes_b": info_b.get("notes", ""),
            "lineage_relation": "Unknown"
        }
        
        # Determine lineage relationship
        lineage = self.metadata["data_lineage"]
        
        # Check if B is a direct descendant of A
        if any(entry["version"] == version_b and entry["parent"] == version_a for entry in lineage):
            comparison["lineage_relation"] = "B is direct descendant of A"
        # Check if A is a direct descendant of B
        elif any(entry["version"] == version_a and entry["parent"] == version_b for entry in lineage):
            comparison["lineage_relation"] = "A is direct descendant of B"
        # Check if they have a common ancestor
        else:
            # Build version ancestry
            ancestry_a = self._get_ancestry(version_a)
            ancestry_b = self._get_ancestry(version_b)
            
            common_ancestors = ancestry_a.intersection(ancestry_b)
            if common_ancestors:
                comparison["lineage_relation"] = f"Common ancestors: {', '.join(common_ancestors)}"
            else:
                comparison["lineage_relation"] = "No common ancestry"
        
        return comparison
    
    def _get_ancestry(self, version: str) -> set:
        """
        Get the ancestry (all parent versions) of a version.
        
        Args:
            version: Version to get ancestry for
            
        Returns:
            Set of ancestor versions
        """
        ancestors = set()
        lineage = self.metadata["data_lineage"]
        
        # Start with the version itself
        current = version
        ancestors.add(current)
        
        # Traverse up the lineage
        while current:
            # Find parent
            parent_entries = [entry for entry in lineage if entry["version"] == current]
            if parent_entries:
                parent = parent_entries[0].get("parent")
                if parent and parent not in ancestors:
                    ancestors.add(parent)
                    current = parent
                else:
                    # No more parents or circular reference
                    break
            else:
                # No parent entry found
                break
        
        return ancestors

# Example usage
if __name__ == "__main__":
    # Initialize version manager
    version_manager = DataVersionManager()
    
    # Print current version
    current_version = version_manager.get_current_version()
    print(f"Current version: {current_version}")
    
    # List all versions
    versions = version_manager.list_versions()
    print(f"Available versions: {versions}")
    
    # Create new version from raw data
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        new_version = version_manager.process_new_data(sys.argv[1])
        print(f"Created new version: {new_version}")
    else:
        print("Usage: python data_versioning.py <path_to_new_data_file>") 