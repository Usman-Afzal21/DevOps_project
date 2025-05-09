"""
Direct Data Analyzer for Banking MLOps

This module analyzes new data directly with RAG without re-vectorizing it.
It provides a more efficient approach for handling new data uploads.
"""

import os
import sys
import json
import logging
import pandas as pd
import datetime
from typing import List, Dict, Any, Optional

# Add project directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rag_pipeline.generator import generate_alert
from embeddings.vector_store import TransactionVectorStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DirectDataAnalyzer:
    """
    Analyzes new banking data directly with existing RAG infrastructure.
    This avoids the need to re-vectorize all data for each upload.
    """
    
    def __init__(
        self,
        alerts_dir: str = "alerts",
        vector_db_dir: str = "data/vector_db",
        alert_threshold: float = 0.7
    ):
        """
        Initialize the direct data analyzer.
        
        Args:
            alerts_dir: Directory to store generated alerts
            vector_db_dir: Directory with vector database (pre-initialized)
            alert_threshold: Threshold for alert generation (0-1)
        """
        self.alerts_dir = os.path.join(os.path.dirname(__file__), '..', alerts_dir)
        self.vector_db_dir = os.path.join(os.path.dirname(__file__), '..', vector_db_dir)
        self.alert_threshold = alert_threshold
        
        # Create alerts directory if it doesn't exist
        os.makedirs(self.alerts_dir, exist_ok=True)
    
    def analyze_new_data(self, data_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze new data directly without vectorizing it.
        
        Args:
            data_df: DataFrame containing the new transaction data
            
        Returns:
            List of generated alerts
        """
        logger.info(f"Analyzing new data with {len(data_df)} records")
        
        # Detect anomalies in the new data
        anomalies = self._detect_anomalies(data_df)
        
        if not anomalies:
            logger.info("No anomalies detected in new data")
            return []
        
        logger.info(f"Detected {len(anomalies)} anomalies in new data")
        
        # Generate alerts for detected anomalies
        alerts = self._generate_alerts_for_anomalies(anomalies, data_df)
        
        # Save alerts
        self._save_alerts(alerts)
        
        return alerts
    
    def _detect_anomalies(self, data_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect anomalies in new data.
        
        Args:
            data_df: DataFrame containing new transaction data
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Skip empty dataframes
        if data_df.empty:
            return anomalies
        
        # Make a copy to avoid modifying the original
        df = data_df.copy()
        
        # 1. Check for large transactions
        if 'TransactionAmount' in df.columns:
            large_threshold = df['TransactionAmount'].quantile(0.95)
            large_txns = df[df['TransactionAmount'] > large_threshold]
            
            if len(large_txns) > 0:
                anomalies.append({
                    'type': 'large_transactions',
                    'severity': 'medium',
                    'count': len(large_txns),
                    'threshold': float(large_threshold),
                    'data': large_txns['TransactionID'].tolist() if 'TransactionID' in large_txns.columns else [],
                    'details': f"Found {len(large_txns)} transactions above {large_threshold:.2f}"
                })
        
        # 2. Check for fraud flags
        if 'FraudFlag' in df.columns:
            # Convert boolean strings to actual booleans if needed
            if df['FraudFlag'].dtype == 'object':
                df['FraudFlag'] = df['FraudFlag'].apply(
                    lambda x: True if str(x).lower() == 'true' else False
                )
                
            fraud_txns = df[df['FraudFlag'] == True]
            
            if len(fraud_txns) > 0:
                anomalies.append({
                    'type': 'fraud_flags',
                    'severity': 'high',
                    'count': len(fraud_txns),
                    'data': fraud_txns['TransactionID'].tolist() if 'TransactionID' in fraud_txns.columns else [],
                    'details': f"Found {len(fraud_txns)} transactions flagged as fraudulent"
                })
        
        # 3. Check for high login attempts
        if 'LoginAttempts' in df.columns:
            login_threshold = 5  # Arbitrary threshold
            high_login_txns = df[df['LoginAttempts'] > login_threshold]
            
            if len(high_login_txns) > 0:
                anomalies.append({
                    'type': 'high_login_attempts',
                    'severity': 'medium',
                    'count': len(high_login_txns),
                    'threshold': login_threshold,
                    'data': high_login_txns['TransactionID'].tolist() if 'TransactionID' in high_login_txns.columns else [],
                    'details': f"Found {len(high_login_txns)} transactions with login attempts > {login_threshold}"
                })
        
        # 4. Check for unusual time patterns
        if 'TransactionDate' in df.columns or 'Date' in df.columns:
            date_col = 'TransactionDate' if 'TransactionDate' in df.columns else 'Date'
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                except:
                    logger.warning(f"Could not convert {date_col} to datetime")
            
            if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                # Get hour of day
                df['hour'] = df[date_col].dt.hour
                
                # Check for overnight transactions (midnight to 5am)
                overnight_txns = df[(df['hour'] >= 0) & (df['hour'] < 5)]
                
                if len(overnight_txns) > 0:
                    anomalies.append({
                        'type': 'overnight_transactions',
                        'severity': 'medium',
                        'count': len(overnight_txns),
                        'data': overnight_txns['TransactionID'].tolist() if 'TransactionID' in overnight_txns.columns else [],
                        'details': f"Found {len(overnight_txns)} transactions during overnight hours (12am-5am)"
                    })
        
        # 5. Look for velocity-based anomalies (rapid succession transactions)
        if 'CustomerID' in df.columns and ('TransactionDate' in df.columns or 'Date' in df.columns):
            date_col = 'TransactionDate' if 'TransactionDate' in df.columns else 'Date'
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                except:
                    logger.warning(f"Could not convert {date_col} to datetime")
            
            if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                # Sort by customer and date
                df = df.sort_values(['CustomerID', date_col])
                
                # Calculate time difference between transactions for same customer
                df['prev_date'] = df.groupby('CustomerID')[date_col].shift(1)
                df['time_diff_minutes'] = (df[date_col] - df['prev_date']).dt.total_seconds() / 60
                
                # Identify rapid succession (< 5 minutes between transactions)
                rapid_txns = df[(df['time_diff_minutes'] < 5) & (df['time_diff_minutes'].notna())]
                
                if len(rapid_txns) > 0:
                    anomalies.append({
                        'type': 'velocity_anomaly',
                        'severity': 'high',
                        'count': len(rapid_txns),
                        'data': rapid_txns['TransactionID'].tolist() if 'TransactionID' in rapid_txns.columns else [],
                        'details': f"Found {len(rapid_txns)} transactions occurring in rapid succession (<5 min apart)"
                    })
        
        # Return all detected anomalies
        return anomalies
    
    def _generate_alerts_for_anomalies(self, anomalies: List[Dict[str, Any]], data_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate alerts for detected anomalies using RAG.
        
        Args:
            anomalies: List of detected anomalies
            data_df: Original data for context
            
        Returns:
            List of generated alerts
        """
        alerts = []
        timestamp = datetime.datetime.now().isoformat()
        
        # Process each anomaly
        for anomaly in anomalies:
            # Initialize alert with basic info
            alert = {
                'id': f"alert_{len(alerts) + 1}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                'timestamp': timestamp,
                'type': anomaly['type'],
                'severity': anomaly['severity'],
                'raw_anomaly': anomaly
            }
            
            # Get transaction IDs involved
            txn_ids = anomaly.get('data', [])
            
            # Add transaction details if available
            if txn_ids and 'TransactionID' in data_df.columns:
                related_txns = data_df[data_df['TransactionID'].isin(txn_ids)]
                
                if not related_txns.empty:
                    # Convert to dict for serialization
                    alert['related_transactions'] = related_txns.head(5).to_dict('records')
            
            # Generate alert content using RAG
            try:
                # Create a query based on anomaly type
                query = self._create_anomaly_query(anomaly)
                
                # Generate alert content using RAG
                # Pass the anomaly data as additional context
                augmented_query = f"{query}\n\nContext: {json.dumps(anomaly)}"
                alert_content = generate_alert(query=augmented_query)
                
                # Add generated content to alert
                alert['title'] = alert_content.get('title', f"Alert: {anomaly['type']}")
                alert['description'] = alert_content.get('description', "Anomaly detected in banking data")
                alert['recommended_actions'] = alert_content.get('recommended_actions', [])
                
                # Add alert to the list
                alerts.append(alert)
                logger.info(f"Generated alert: {alert['title']}")
                
            except Exception as e:
                logger.error(f"Error generating alert for anomaly {anomaly['type']}: {e}")
                # Create a basic alert without RAG
                alert['title'] = f"Alert: {anomaly['type']}"
                alert['description'] = f"Anomaly detected of type {anomaly['type']} with severity {anomaly['severity']}"
                alert['recommended_actions'] = ["Review the anomaly details", "Investigate transactions"]
                alerts.append(alert)
        
        return alerts
    
    def _create_anomaly_query(self, anomaly: Dict[str, Any]) -> str:
        """
        Create a query for the RAG system based on anomaly type.
        
        Args:
            anomaly: Anomaly details
            
        Returns:
            Query string for RAG
        """
        anomaly_type = anomaly['type']
        
        if anomaly_type == 'large_transactions':
            return f"Generate an alert for {anomaly['count']} unusually large transactions above {anomaly['threshold']:.2f}"
            
        elif anomaly_type == 'fraud_flags':
            return f"Generate an alert for {anomaly['count']} transactions flagged as potential fraud"
            
        elif anomaly_type == 'high_login_attempts':
            return f"Generate an alert for {anomaly['count']} transactions with high login attempts above {anomaly['threshold']}"
            
        elif anomaly_type == 'overnight_transactions':
            return f"Generate an alert for {anomaly['count']} transactions occurring during overnight hours (12am-5am)"
            
        elif anomaly_type == 'velocity_anomaly':
            return f"Generate an alert for {anomaly['count']} transactions occurring in rapid succession"
            
        else:
            # Generic query for any other type
            return f"Generate an alert for {anomaly_type} with severity {anomaly['severity']}"
    
    def _save_alerts(self, alerts: List[Dict[str, Any]]) -> str:
        """
        Save generated alerts to a file.
        
        Args:
            alerts: List of alerts to save
            
        Returns:
            Path to the saved alerts file
        """
        if not alerts:
            logger.info("No alerts to save")
            return ""
        
        # Create filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        alerts_path = os.path.join(self.alerts_dir, f"alerts_{timestamp}.json")
        
        # Save alerts to file
        with open(alerts_path, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        logger.info(f"Saved {len(alerts)} alerts to {alerts_path}")
        return alerts_path

# Direct function to analyze new data from CSV
def analyze_csv_data(csv_file_path: str) -> List[Dict[str, Any]]:
    """
    Analyze new data from a CSV file without vectorizing it.
    
    Args:
        csv_file_path: Path to the CSV file with new data
        
    Returns:
        List of generated alerts
    """
    logger.info(f"Analyzing new data from CSV: {csv_file_path}")
    
    try:
        # Load the CSV data
        data_df = pd.read_csv(csv_file_path)
        
        # Initialize the analyzer
        analyzer = DirectDataAnalyzer()
        
        # Analyze the data
        alerts = analyzer.analyze_new_data(data_df)
        
        logger.info(f"Generated {len(alerts)} alerts from {csv_file_path}")
        return alerts
    except Exception as e:
        logger.error(f"Error analyzing CSV data: {e}")
        return []

# Run standalone if executed directly
if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        alerts = analyze_csv_data(csv_file)
        print(f"Generated {len(alerts)} alerts from {csv_file}")
    else:
        print("Usage: python direct_data_analyzer.py <csv_file>")
        sys.exit(1) 