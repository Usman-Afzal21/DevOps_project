"""
Automatic Alert Generator Service

This module automatically analyzes banking data and generates alerts
without manual prompting. It integrates with the RAG pipeline and runs
after data updates to identify issues requiring attention.
"""

import os
import sys
import json
import logging
import datetime
import pandas as pd
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

class AutomaticAlertGenerator:
    """
    Automatically analyzes banking data and generates relevant alerts.
    """
    
    def __init__(
        self,
        alerts_dir: str = "alerts",
        processed_data_dir: str = "data/processed",
        vector_db_dir: str = "data/vector_db",
        alert_threshold: float = 0.7
    ):
        """
        Initialize the automatic alert generator.
        
        Args:
            alerts_dir: Directory to store generated alerts
            processed_data_dir: Directory with processed data
            vector_db_dir: Directory with vector database
            alert_threshold: Threshold for alert generation (0-1)
        """
        self.alerts_dir = os.path.join(os.path.dirname(__file__), '..', alerts_dir)
        self.processed_data_dir = os.path.join(os.path.dirname(__file__), '..', processed_data_dir)
        self.vector_db_dir = os.path.join(os.path.dirname(__file__), '..', vector_db_dir)
        self.alert_threshold = alert_threshold
        
        # Create alerts directory if it doesn't exist
        os.makedirs(self.alerts_dir, exist_ok=True)
        
        # Initialize vector store for RAG
        self.vector_store = TransactionVectorStore(persist_directory=self.vector_db_dir)
    
    def load_processed_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all processed data files.
        
        Returns:
            Dictionary of DataFrames
        """
        logger.info(f"Loading processed data from {self.processed_data_dir}")
        
        data_dict = {}
        
        try:
            # Find all processed CSV files
            csv_files = [f for f in os.listdir(self.processed_data_dir) 
                         if f.endswith('.csv') and f.startswith('processed_')]
            
            # Load each file
            for filename in csv_files:
                file_path = os.path.join(self.processed_data_dir, filename)
                dataset_name = os.path.splitext(filename)[0].replace('processed_', '')
                
                try:
                    df = pd.read_csv(file_path)
                    data_dict[dataset_name] = df
                    logger.info(f"Loaded {dataset_name}: {len(df)} records")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
            
            # Also load branch summaries
            summary_path = os.path.join(self.processed_data_dir, 'branch_summaries.csv')
            if os.path.exists(summary_path):
                try:
                    df = pd.read_csv(summary_path)
                    data_dict['branch_summaries'] = df
                    logger.info(f"Loaded branch_summaries: {len(df)} records")
                except Exception as e:
                    logger.error(f"Error loading branch_summaries.csv: {e}")
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return {}
    
    def detect_anomalies(self, data_dict: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the data that require alerts.
        
        Args:
            data_dict: Dictionary of DataFrames
            
        Returns:
            List of anomaly details
        """
        logger.info("Detecting anomalies in processed data")
        
        anomalies = []
        
        # Check transactions data for anomalies
        if 'transactions' in data_dict or 'bank_transactions_data_2' in data_dict:
            transactions_df = data_dict.get('transactions', data_dict.get('bank_transactions_data_2'))
            
            # Large transaction anomalies
            if 'TransactionAmount' in transactions_df.columns:
                large_txn_threshold = transactions_df['TransactionAmount'].quantile(0.99)
                large_txns = transactions_df[transactions_df['TransactionAmount'] > large_txn_threshold]
                
                if len(large_txns) > 0:
                    # Get available columns for data extraction
                    available_cols = ['TransactionID', 'TransactionAmount', 'TransactionType']
                    if 'BranchID' in transactions_df.columns:
                        available_cols.append('BranchID')
                    elif 'Location' in transactions_df.columns:
                        available_cols.append('Location')
                        
                    # Only include columns that exist in the dataframe
                    data_cols = [col for col in available_cols if col in large_txns.columns]
                    
                    anomalies.append({
                        'type': 'large_transactions',
                        'severity': 'medium',
                        'count': len(large_txns),
                        'threshold': large_txn_threshold,
                        'data': large_txns[data_cols].head(10).to_dict('records'),
                        'timestamp': datetime.datetime.now().isoformat()
                    })
            
            # Fraud flags
            if 'FraudFlag' in transactions_df.columns:
                fraud_txns = transactions_df[transactions_df['FraudFlag'] == True]
                
                if len(fraud_txns) > 0:
                    # Get available columns for data extraction
                    available_cols = ['TransactionID', 'TransactionAmount', 'TransactionType']
                    if 'BranchID' in transactions_df.columns:
                        available_cols.append('BranchID')
                    elif 'Location' in transactions_df.columns:
                        available_cols.append('Location')
                        
                    # Only include columns that exist in the dataframe
                    data_cols = [col for col in available_cols if col in fraud_txns.columns]
                    
                    anomalies.append({
                        'type': 'fraud_flags',
                        'severity': 'high',
                        'count': len(fraud_txns),
                        'data': fraud_txns[data_cols].head(10).to_dict('records'),
                        'timestamp': datetime.datetime.now().isoformat()
                    })
            
            # Login attempts anomalies
            if 'LoginAttempts' in transactions_df.columns:
                high_login_threshold = 3
                high_login_txns = transactions_df[transactions_df['LoginAttempts'] > high_login_threshold]
                
                if len(high_login_txns) > 0:
                    # Get available columns for data extraction
                    available_cols = ['TransactionID', 'LoginAttempts']
                    if 'CustomerID' in transactions_df.columns:
                        available_cols.append('CustomerID')
                    
                    # Only include columns that exist in the dataframe
                    data_cols = [col for col in available_cols if col in high_login_txns.columns]
                    
                    anomalies.append({
                        'type': 'high_login_attempts',
                        'severity': 'medium',
                        'count': len(high_login_txns),
                        'threshold': high_login_threshold,
                        'data': high_login_txns[data_cols].head(10).to_dict('records'),
                        'timestamp': datetime.datetime.now().isoformat()
                    })
        
        # Check branch summaries for anomalies
        if 'branch_summaries' in data_dict:
            summaries_df = data_dict['branch_summaries']
            
            # Branches with high large transaction ratio
            if 'LargeTransactionCount' in summaries_df.columns and 'TransactionCount' in summaries_df.columns:
                summaries_df['LargeTransactionRatio'] = summaries_df['LargeTransactionCount'] / summaries_df['TransactionCount']
                high_ratio_threshold = 0.15
                high_ratio_branches = summaries_df[summaries_df['LargeTransactionRatio'] > high_ratio_threshold]
                
                if len(high_ratio_branches) > 0:
                    # Get available columns for data extraction
                    available_cols = []
                    if 'Location' in summaries_df.columns:
                        available_cols.append('Location')
                    elif 'BranchID' in summaries_df.columns:
                        available_cols.append('BranchID')
                    
                    if 'Date' in summaries_df.columns:
                        available_cols.append('Date')
                    
                    available_cols.extend(['LargeTransactionRatio', 'TransactionCount'])
                    
                    # Only include columns that exist in the dataframe
                    data_cols = [col for col in available_cols if col in high_ratio_branches.columns]
                    
                    anomalies.append({
                        'type': 'high_large_transaction_ratio',
                        'severity': 'medium',
                        'count': len(high_ratio_branches),
                        'threshold': high_ratio_threshold,
                        'data': high_ratio_branches[data_cols].head(10).to_dict('records'),
                        'timestamp': datetime.datetime.now().isoformat()
                    })
            
            # Branches with unusual activity patterns
            if 'TransactionCount' in summaries_df.columns:
                # Determine the grouping column (Location or BranchID)
                group_col = 'Location' if 'Location' in summaries_df.columns else 'BranchID' if 'BranchID' in summaries_df.columns else None
                
                if group_col:
                    grouped = summaries_df.groupby(group_col)['TransactionCount'].agg(['mean', 'std']).reset_index()
                    
                    for _, row in grouped.iterrows():
                        location = row[group_col]
                        mean_count = row['mean']
                        std_count = row['std'] if not pd.isna(row['std']) else 1
                        
                        for _, summary_row in summaries_df[summaries_df[group_col] == location].iterrows():
                            if abs(summary_row['TransactionCount'] - mean_count) > 2 * std_count:
                                # Determine date field
                                date_field = 'Date' if 'Date' in summary_row and not pd.isna(summary_row['Date']) else None
                                date_value = str(summary_row['Date']) if date_field else 'Unknown'
                                
                                anomalies.append({
                                    'type': 'unusual_transaction_volume',
                                    'severity': 'low',
                                    'branch': location,
                                    'date': date_value,
                                    'actual_count': int(summary_row['TransactionCount']),
                                    'expected_count': int(mean_count),
                                    'std_deviation': float(std_count),
                                    'timestamp': datetime.datetime.now().isoformat()
                                })
                                break  # Only one anomaly per branch
        
        # Process other datasets based on banking domain knowledge
        for dataset_name, df in data_dict.items():
            if dataset_name in ['transactions', 'bank_transactions_data_2', 'branch_summaries']:
                continue  # Already processed
                
            # For other datasets, look for common patterns that might indicate issues
            
            # Check for any columns that might contain flags
            flag_cols = [col for col in df.columns if 'flag' in col.lower() or 'alert' in col.lower()]
            for col in flag_cols:
                try:
                    if df[col].dtype == bool:
                        flagged_rows = df[df[col] == True]
                    else:
                        flagged_rows = df[df[col] > 0]
                    
                    if len(flagged_rows) > 0:
                        anomalies.append({
                            'type': f'{dataset_name}_{col}',
                            'severity': 'medium',
                            'count': len(flagged_rows),
                            'data': flagged_rows.head(10).to_dict('records'),
                            'timestamp': datetime.datetime.now().isoformat()
                        })
                except Exception as e:
                    logger.warning(f"Error processing flag column {col} in {dataset_name}: {e}")
                    
            # Check for extreme values in numerical columns
            num_cols = df.select_dtypes(include=['number']).columns
            for col in num_cols:
                try:
                    if 'id' in col.lower() or 'index' in col.lower():
                        continue  # Skip ID columns
                        
                    q95 = df[col].quantile(0.95)
                    q05 = df[col].quantile(0.05)
                    extreme_high = df[df[col] > q95 + 3 * (q95 - q05)]
                    extreme_low = df[df[col] < q05 - 3 * (q95 - q05)]
                    
                    if len(extreme_high) > df.shape[0] * 0.01:  # If more than 1% of data
                        anomalies.append({
                            'type': f'{dataset_name}_{col}_extreme_high',
                            'severity': 'low',
                            'count': len(extreme_high),
                            'threshold': float(q95 + 3 * (q95 - q05)),
                            'timestamp': datetime.datetime.now().isoformat()
                        })
                    
                    if len(extreme_low) > df.shape[0] * 0.01:  # If more than 1% of data
                        anomalies.append({
                            'type': f'{dataset_name}_{col}_extreme_low',
                            'severity': 'low',
                            'count': len(extreme_low),
                            'threshold': float(q05 - 3 * (q95 - q05)),
                            'timestamp': datetime.datetime.now().isoformat()
                        })
                except Exception as e:
                    logger.warning(f"Error analyzing column {col} in {dataset_name}: {e}")
        
        logger.info(f"Detected {len(anomalies)} anomalies across datasets")
        return anomalies
    
    def generate_alerts(self, anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate detailed alerts based on detected anomalies.
        
        Args:
            anomalies: List of anomaly details
            
        Returns:
            List of generated alerts
        """
        if not anomalies:
            logger.info("No anomalies detected, no alerts generated")
            return []
        
        logger.info(f"Generating alerts for {len(anomalies)} anomalies")
        alerts = []
        
        for anomaly in anomalies:
            # Skip low severity anomalies if they don't meet threshold
            if anomaly['severity'] == 'low' and self.alert_threshold > 0.5:
                continue
                
            # Create base alert details
            alert = {
                'id': f"alert_{len(alerts) + 1}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                'type': anomaly['type'],
                'severity': anomaly['severity'],
                'timestamp': datetime.datetime.now().isoformat(),
                'status': 'new',
                'anomaly_data': anomaly
            }
            
            # Generate alert content using RAG
            try:
                # Create a query based on anomaly type
                query = self._create_anomaly_query(anomaly)
                
                # Generate alert content using RAG
                # Pass the query as additional context instead of as a parameter
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
            
        elif anomaly_type == 'high_large_transaction_ratio':
            return f"Generate an alert for {anomaly['count']} branches with unusually high ratio of large transactions"
            
        elif anomaly_type == 'unusual_transaction_volume':
            return f"Generate an alert for branch {anomaly['branch']} with unusual transaction volume of {anomaly['actual_count']} vs expected {anomaly['expected_count']}"
            
        elif 'extreme_high' in anomaly_type:
            field = anomaly_type.replace('_extreme_high', '')
            return f"Generate an alert for unusually high values in {field}"
            
        elif 'extreme_low' in anomaly_type:
            field = anomaly_type.replace('_extreme_low', '')
            return f"Generate an alert for unusually low values in {field}"
            
        else:
            # Generic query for any other type
            return f"Generate an alert for {anomaly_type} with severity {anomaly['severity']}"
    
    def save_alerts(self, alerts: List[Dict[str, Any]]) -> str:
        """
        Save generated alerts to a file.
        
        Args:
            alerts: List of alerts
            
        Returns:
            Path to the alerts file
        """
        if not alerts:
            logger.info("No alerts to save")
            return ""
        
        try:
            # Create filename with timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"alerts_{timestamp}.json"
            file_path = os.path.join(self.alerts_dir, filename)
            
            # Save alerts to file
            with open(file_path, 'w') as f:
                json.dump(alerts, f, indent=2)
                
            logger.info(f"Saved {len(alerts)} alerts to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving alerts: {e}")
            return ""
    
    def load_latest_alerts(self) -> List[Dict[str, Any]]:
        """
        Load the latest generated alerts.
        
        Returns:
            List of alerts
        """
        try:
            # Find the most recent alerts file
            alert_files = [f for f in os.listdir(self.alerts_dir) if f.startswith('alerts_') and f.endswith('.json')]
            
            if not alert_files:
                logger.info("No alert files found")
                return []
                
            # Sort by timestamp (which is part of the filename)
            alert_files.sort(reverse=True)
            latest_file = os.path.join(self.alerts_dir, alert_files[0])
            
            # Load alerts
            with open(latest_file, 'r') as f:
                alerts = json.load(f)
                
            logger.info(f"Loaded {len(alerts)} alerts from {latest_file}")
            return alerts
            
        except Exception as e:
            logger.error(f"Error loading latest alerts: {e}")
            return []
    
    def run(self) -> List[Dict[str, Any]]:
        """
        Run the complete alert generation process.
        
        Returns:
            List of generated alerts
        """
        logger.info("Starting automatic alert generation process")
        
        # Load processed data
        data_dict = self.load_processed_data()
        
        if not data_dict:
            logger.warning("No processed data found, cannot generate alerts")
            return []
        
        # Detect anomalies
        anomalies = self.detect_anomalies(data_dict)
        
        if not anomalies:
            logger.info("No anomalies detected, no alerts needed")
            return []
        
        # Generate alerts
        alerts = self.generate_alerts(anomalies)
        
        # Save alerts
        self.save_alerts(alerts)
        
        logger.info(f"Generated {len(alerts)} alerts")
        return alerts

# Run alert generator if executed directly
if __name__ == "__main__":
    alert_generator = AutomaticAlertGenerator()
    alerts = alert_generator.run()
    print(f"Generated {len(alerts)} alerts") 