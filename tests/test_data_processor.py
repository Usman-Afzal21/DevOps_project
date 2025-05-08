"""
Tests for the data processing module.

This module contains unit tests for the TransactionDataProcessor class.
"""

import os
import sys
import unittest
import pandas as pd
import tempfile
from datetime import datetime

# Add parent directory to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.processed.data_processor import TransactionDataProcessor

class TestDataProcessor(unittest.TestCase):
    """Tests for the TransactionDataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Initialize processor with paths to test data
        self.processor = TransactionDataProcessor(
            raw_data_path=os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'),
            processed_data_path=self.output_dir
        )
        
        # Create a small test DataFrame
        self.test_data = pd.DataFrame({
            'TransactionID': ['TX000001', 'TX000002', 'TX000003'],
            'AccountID': ['AC00001', 'AC00002', 'AC00003'],
            'TransactionAmount': [100.0, 200.0, 300.0],
            'TransactionDate': ['2023-01-01 10:00:00', '2023-01-02 11:00:00', '2023-01-03 12:00:00'],
            'TransactionType': ['Debit', 'Credit', 'Debit'],
            'Location': ['New York', 'Chicago', 'New York'],
            'DeviceID': ['D0001', 'D0002', 'D0003'],
            'IP Address': ['1.1.1.1', '2.2.2.2', '3.3.3.3'],
            'MerchantID': ['M001', 'M002', 'M003'],
            'Channel': ['ATM', 'Online', 'Branch'],
            'CustomerAge': [25, 35, 45],
            'CustomerOccupation': ['Student', 'Engineer', 'Doctor'],
            'TransactionDuration': [60, 120, 180],
            'LoginAttempts': [1, 2, 1],
            'AccountBalance': [1000.0, 2000.0, 3000.0],
            'PreviousTransactionDate': ['2022-12-31 09:00:00', '2022-12-31 10:00:00', '2022-12-31 11:00:00']
        })
        
        # Convert date columns to datetime
        self.test_data['TransactionDate'] = pd.to_datetime(self.test_data['TransactionDate'])
        self.test_data['PreviousTransactionDate'] = pd.to_datetime(self.test_data['PreviousTransactionDate'])
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        # Set test data in processor
        self.processor.data = self.test_data.copy()
        
        # Clean the data
        cleaned_data = self.processor.clean_data()
        
        # Check that dates are datetime objects
        self.assertTrue(pd.api.types.is_datetime64_dtype(cleaned_data['TransactionDate']))
        self.assertTrue(pd.api.types.is_datetime64_dtype(cleaned_data['PreviousTransactionDate']))
        
        # Check that all rows were preserved (no duplicates in test data)
        self.assertEqual(len(cleaned_data), len(self.test_data))
    
    def test_create_features(self):
        """Test feature creation functionality."""
        # Set test data in processor
        self.processor.data = self.test_data.copy()
        
        # Create features
        processed_data = self.processor.create_features()
        
        # Check that new features were created
        expected_new_features = [
            'TransactionDay', 'TransactionMonth', 'TransactionYear',
            'TransactionDayOfWeek', 'IsWeekend', 'DaysSincePreviousTx',
            'IsLargeTransaction', 'IsCredit', 'AgeGroup'
        ]
        
        for feature in expected_new_features:
            self.assertIn(feature, processed_data.columns)
        
        # Check specific feature values
        self.assertEqual(processed_data.loc[1, 'IsCredit'], 1)  # Second row is Credit
        self.assertEqual(processed_data.loc[0, 'IsCredit'], 0)  # First row is Debit
    
    def test_generate_branch_summaries(self):
        """Test branch summary generation."""
        # Set test data in processor
        self.processor.data = self.test_data.copy()
        self.processor.create_features()
        
        # Generate branch summaries
        summaries = self.processor.generate_branch_summaries()
        
        # Check that summaries were generated
        self.assertGreater(len(summaries), 0)
        
        # Check that summaries have expected columns
        expected_columns = [
            'Location', 'Date', 'TransactionCount', 'TotalAmount',
            'AverageAmount', 'CreditTransactionCount', 'LargeTransactionCount'
        ]
        
        for column in expected_columns:
            self.assertIn(column, summaries.columns)
        
        # Check that New York has 2 transactions
        ny_summary = summaries[summaries['Location'] == 'New York']
        if not ny_summary.empty:
            self.assertEqual(ny_summary['TransactionCount'].iloc[0], 2)

if __name__ == '__main__':
    unittest.main() 