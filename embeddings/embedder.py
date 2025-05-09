"""
Embeddings Generator for Banking Transaction Data.

This module handles:
- Converting transaction data to text
- Generating embeddings for these texts
- Preparing data for the vector store
"""

import os
import pandas as pd
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TransactionEmbedder:
    """
    Generate embeddings from transaction data for RAG.
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the transaction embedder.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def transaction_to_text(self, transaction: Dict[str, Any]) -> str:
        """
        Convert a transaction record to a descriptive text format for embedding.
        
        Args:
            transaction: Dictionary representing a transaction
            
        Returns:
            String representation of the transaction
        """
        # Format the transaction as a descriptive text
        text = (
            f"Transaction {transaction['TransactionID']} was a {transaction['TransactionAmount']:.2f} "
            f"{transaction['TransactionType']} transaction on {transaction['TransactionDate']} "
            f"at {transaction['Location']} branch via {transaction['Channel']}. "
            f"The customer is {transaction['CustomerAge']} years old and works as a {transaction['CustomerOccupation']}. "
            f"The transaction took {transaction['TransactionDuration']} seconds with {transaction['LoginAttempts']} login attempts. "
            f"Account balance after transaction was {transaction['AccountBalance']:.2f}."
        )
        return text
    
    def branch_summary_to_text(self, summary: Dict[str, Any]) -> str:
        """
        Convert a branch summary record to a descriptive text format for embedding.
        
        Args:
            summary: Dictionary representing a branch summary
            
        Returns:
            String representation of the branch summary
        """
        # Format the branch summary as a descriptive text
        text = (
            f"Daily summary for {summary.get('Location_', 'Unknown')} on {summary.get('Date_', 'Unknown')}: "
            f"Processed {summary.get('TransactionCount', 0)} transactions totaling {summary.get('TotalAmount', 0):.2f}. "
            f"Average transaction amount was {summary.get('AverageAmount', 0):.2f} with standard deviation of {summary.get('AmountStdDev', 0):.2f}. "
            f"The branch had {summary.get('CreditTransactionCount', 0)} credit transactions and {summary.get('LargeTransactionCount', 0)} large transactions. "
            f"Average login attempts were {summary.get('AverageLoginAttempts', 0):.2f} with maximum of {summary.get('MaxLoginAttempts', 0)}. "
        )
        
        # Conditionally add customer age information if available
        if 'AverageCustomerAge' in summary:
            text += f"Average customer age was {summary.get('AverageCustomerAge', 0):.2f} years. "
        
        # Conditionally add transaction duration if available
        if 'AverageTransactionDuration' in summary:
            text += f"Average transaction duration was {summary.get('AverageTransactionDuration', 0):.2f} seconds. "
        
        # Conditionally add fraud risk score if available
        if 'FraudRiskScore' in summary:
            text += f"Fraud risk score: {summary.get('FraudRiskScore', 0):.4f}."
        else:
            text += f"Fraud risk score: {0.01:.4f}."
        
        return text
    
    def generate_transaction_embeddings(self, transactions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate embeddings for transaction records.
        
        Args:
            transactions_df: DataFrame of transaction records
            
        Returns:
            Dictionary with texts and their corresponding embeddings
        """
        logger.info(f"Generating embeddings for {len(transactions_df)} transactions")
        
        # Convert transactions to texts
        texts = []
        ids = []
        metadata = []
        
        for _, transaction in transactions_df.iterrows():
            transaction_dict = transaction.to_dict()
            text = self.transaction_to_text(transaction_dict)
            texts.append(text)
            ids.append(transaction_dict['TransactionID'])
            metadata.append({
                'id': transaction_dict['TransactionID'],
                'date': str(transaction_dict['TransactionDate']),
                'location': transaction_dict['Location'],
                'type': transaction_dict['TransactionType'],
                'amount': float(transaction_dict['TransactionAmount']),
                'channel': transaction_dict['Channel']
            })
        
        # Generate embeddings
        embeddings = self.model.encode(texts)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        return {
            'texts': texts,
            'embeddings': embeddings,
            'ids': ids,
            'metadata': metadata
        }
    
    def generate_summary_embeddings(self, summaries_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate embeddings for branch summary records.
        
        Args:
            summaries_df: DataFrame of branch summary records
            
        Returns:
            Dictionary with texts and their corresponding embeddings
        """
        logger.info(f"Generating embeddings for {len(summaries_df)} branch summaries")
        
        # Convert summaries to texts
        texts = []
        ids = []
        metadata = []
        
        for idx, summary in summaries_df.iterrows():
            summary_dict = summary.to_dict()
            text = self.branch_summary_to_text(summary_dict)
            texts.append(text)
            ids.append(f"summary_{idx}")
            
            # Create metadata with safe access to values
            meta_entry = {
                'id': f"summary_{idx}",
                'date': str(summary_dict.get('Date_', '')),
                'location': summary_dict.get('Location_', 'Unknown'),
                'transaction_count': int(summary_dict.get('TransactionCount', 0))
            }
            
            # Only add fraud_risk_score if it exists
            if 'FraudRiskScore' in summary_dict:
                meta_entry['fraud_risk_score'] = float(summary_dict.get('FraudRiskScore', 0.01))
            else:
                meta_entry['fraud_risk_score'] = 0.01
            
            metadata.append(meta_entry)
        
        # Generate embeddings
        embeddings = self.model.encode(texts)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        return {
            'texts': texts,
            'embeddings': embeddings,
            'ids': ids,
            'metadata': metadata
        }
    
    def save_embeddings(self, embeddings_dict: Dict[str, Any], output_dir: str, prefix: str) -> str:
        """
        Save embeddings and related data to files.
        
        Args:
            embeddings_dict: Dictionary containing texts, embeddings, ids, and metadata
            output_dir: Directory to save the embeddings
            prefix: Prefix for the output files
            
        Returns:
            Path to the saved embeddings file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save texts
        texts_file = os.path.join(output_dir, f"{prefix}_texts.txt")
        with open(texts_file, 'w') as f:
            for text in embeddings_dict['texts']:
                f.write(f"{text}\n")
        
        # Save embeddings
        embeddings_file = os.path.join(output_dir, f"{prefix}_embeddings.npy")
        np.save(embeddings_file, embeddings_dict['embeddings'])
        
        # Save metadata
        metadata_file = os.path.join(output_dir, f"{prefix}_metadata.csv")
        pd.DataFrame(embeddings_dict['metadata']).to_csv(metadata_file, index=False)
        
        logger.info(f"Saved embeddings to {embeddings_file}")
        
        return embeddings_file
    
    def process_and_embed(self, transactions_path: str, summaries_path: str, output_dir: str) -> Dict[str, str]:
        """
        Process transaction and summary data and generate embeddings.
        
        Args:
            transactions_path: Path to the processed transactions file
            summaries_path: Path to the branch summaries file
            output_dir: Directory to save the embeddings
            
        Returns:
            Dictionary with paths to the saved embedding files
        """
        logger.info("Starting embedding generation")
        
        # Load the processed data
        transactions_df = pd.read_csv(transactions_path)
        summaries_df = pd.read_csv(summaries_path)
        
        # Convert date columns to datetime
        transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
        transactions_df['PreviousTransactionDate'] = pd.to_datetime(transactions_df['PreviousTransactionDate'])
        
        # Generate embeddings
        transaction_embeddings = self.generate_transaction_embeddings(transactions_df)
        summary_embeddings = self.generate_summary_embeddings(summaries_df)
        
        # Save embeddings
        transaction_embeddings_file = self.save_embeddings(
            transaction_embeddings, output_dir, "transactions"
        )
        summary_embeddings_file = self.save_embeddings(
            summary_embeddings, output_dir, "summaries"
        )
        
        return {
            'transaction_embeddings': transaction_embeddings_file,
            'summary_embeddings': summary_embeddings_file
        }

if __name__ == "__main__":
    # Example usage
    embedder = TransactionEmbedder()
    output_files = embedder.process_and_embed(
        transactions_path="data/processed/processed_transactions.csv",
        summaries_path="data/processed/branch_summaries.csv",
        output_dir="embeddings"
    )
    print(f"Embedding generation complete. Output files: {output_files}")

