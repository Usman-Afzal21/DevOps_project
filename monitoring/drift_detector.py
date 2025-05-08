"""
Model Drift Detection Module.

This module handles:
- Detecting data and concept drift
- Monitoring retrieval quality
- Tracking embedding space changes
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, average_precision_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DriftDetector:
    """
    Detect model drift and performance degradation.
    """
    
    def __init__(self, log_dir: str = "monitoring/logs", plots_dir: str = "monitoring/plots"):
        """
        Initialize the drift detector.
        
        Args:
            log_dir: Directory containing log files
            plots_dir: Directory to save plots
        """
        self.log_dir = log_dir
        self.plots_dir = plots_dir
        
        # Create directories if they don't exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Log file paths
        self.performance_log_file = os.path.join(log_dir, "performance_metrics.jsonl")
        self.usage_log_file = os.path.join(log_dir, "usage_metrics.jsonl")
        self.feedback_log_file = os.path.join(log_dir, "feedback_metrics.jsonl")
        
        # Embedding samples file for distribution monitoring
        self.embedding_samples_file = os.path.join(log_dir, "embedding_samples.npz")
    
    def load_performance_data(self) -> pd.DataFrame:
        """
        Load performance data from log file.
        
        Returns:
            DataFrame with performance metrics
        """
        if not os.path.exists(self.performance_log_file):
            logger.warning("Performance log file does not exist")
            return pd.DataFrame()
        
        try:
            df = pd.read_json(self.performance_log_file, lines=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
            return pd.DataFrame()
    
    def load_feedback_data(self) -> pd.DataFrame:
        """
        Load feedback data from log file.
        
        Returns:
            DataFrame with feedback metrics
        """
        if not os.path.exists(self.feedback_log_file):
            logger.warning("Feedback log file does not exist")
            return pd.DataFrame()
        
        try:
            df = pd.read_json(self.feedback_log_file, lines=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")
            return pd.DataFrame()
    
    def detect_response_time_drift(self, 
                                  baseline_days: int = 30, 
                                  current_days: int = 7,
                                  threshold_percent: float = 20.0) -> Dict[str, Any]:
        """
        Detect drift in response time.
        
        Args:
            baseline_days: Number of days to use as baseline
            current_days: Number of days to compare against baseline
            threshold_percent: Threshold percentage for drift detection
            
        Returns:
            Dictionary with drift detection results
        """
        df = self.load_performance_data()
        
        if df.empty:
            return {"drift_detected": False, "reason": "No data available"}
        
        # Define baseline and current periods
        now = pd.Timestamp.now()
        baseline_start = now - timedelta(days=baseline_days)
        current_start = now - timedelta(days=current_days)
        
        baseline_df = df[(df['timestamp'] >= baseline_start) & (df['timestamp'] < current_start)]
        current_df = df[df['timestamp'] >= current_start]
        
        if baseline_df.empty or current_df.empty:
            return {"drift_detected": False, "reason": "Insufficient data"}
        
        # Calculate metrics
        baseline_response_time = baseline_df['response_time_seconds'].mean()
        current_response_time = current_df['response_time_seconds'].mean()
        
        # Check for drift
        response_time_change = ((current_response_time - baseline_response_time) / baseline_response_time) * 100
        
        drift_detected = abs(response_time_change) > threshold_percent
        
        result = {
            "drift_detected": drift_detected,
            "metric": "response_time_seconds",
            "baseline_value": baseline_response_time,
            "current_value": current_response_time,
            "percent_change": response_time_change,
            "threshold": threshold_percent
        }
        
        # Generate plot
        if drift_detected:
            self._plot_response_time_trend(df, baseline_start, current_start, result)
        
        return result
    
    def detect_feedback_drift(self, 
                            baseline_days: int = 30, 
                            current_days: int = 7,
                            threshold_percent: float = 15.0) -> Dict[str, Any]:
        """
        Detect drift in user feedback quality ratings.
        
        Args:
            baseline_days: Number of days to use as baseline
            current_days: Number of days to compare against baseline
            threshold_percent: Threshold percentage for drift detection
            
        Returns:
            Dictionary with drift detection results
        """
        df = self.load_feedback_data()
        
        if df.empty:
            return {"drift_detected": False, "reason": "No data available"}
        
        # Define baseline and current periods
        now = pd.Timestamp.now()
        baseline_start = now - timedelta(days=baseline_days)
        current_start = now - timedelta(days=current_days)
        
        baseline_df = df[(df['timestamp'] >= baseline_start) & (df['timestamp'] < current_start)]
        current_df = df[df['timestamp'] >= current_start]
        
        if baseline_df.empty or current_df.empty:
            return {"drift_detected": False, "reason": "Insufficient data"}
        
        # Calculate metrics
        baseline_quality = baseline_df['alert_quality'].mean()
        current_quality = current_df['alert_quality'].mean()
        
        # Check for drift (negative change is concerning)
        quality_change = ((current_quality - baseline_quality) / baseline_quality) * 100
        
        drift_detected = quality_change < -threshold_percent
        
        result = {
            "drift_detected": drift_detected,
            "metric": "alert_quality",
            "baseline_value": baseline_quality,
            "current_value": current_quality,
            "percent_change": quality_change,
            "threshold": -threshold_percent
        }
        
        # Generate plot
        if drift_detected:
            self._plot_feedback_trend(df, baseline_start, current_start, result)
        
        return result
    
    def detect_embedding_drift(self, 
                             baseline_embeddings: np.ndarray,
                             current_embeddings: np.ndarray,
                             threshold_distance: float = 0.1) -> Dict[str, Any]:
        """
        Detect drift in embedding space using cosine distance.
        
        Args:
            baseline_embeddings: Previous embeddings
            current_embeddings: Current embeddings
            threshold_distance: Threshold for average cosine distance
            
        Returns:
            Dictionary with drift detection results
        """
        if baseline_embeddings.shape[0] == 0 or current_embeddings.shape[0] == 0:
            return {"drift_detected": False, "reason": "Insufficient data"}
        
        # Calculate average embedding vectors
        baseline_avg = np.mean(baseline_embeddings, axis=0)
        current_avg = np.mean(current_embeddings, axis=0)
        
        # Calculate cosine distance
        distance = cosine(baseline_avg, current_avg)
        
        drift_detected = distance > threshold_distance
        
        result = {
            "drift_detected": drift_detected,
            "metric": "embedding_cosine_distance",
            "distance": distance,
            "threshold": threshold_distance
        }
        
        # Generate plot
        if drift_detected:
            self._plot_embedding_drift(baseline_embeddings, current_embeddings, result)
        
        return result
    
    def _plot_response_time_trend(self, 
                                 df: pd.DataFrame, 
                                 baseline_start: datetime, 
                                 current_start: datetime,
                                 result: Dict[str, Any]) -> None:
        """
        Plot response time trend.
        
        Args:
            df: DataFrame with performance metrics
            baseline_start: Start of baseline period
            current_start: Start of current period
            result: Drift detection results
        """
        plt.figure(figsize=(12, 6))
        
        # Group by day and calculate average response time
        df['date'] = df['timestamp'].dt.date
        daily_avg = df.groupby('date')['response_time_seconds'].mean().reset_index()
        daily_avg['date'] = pd.to_datetime(daily_avg['date'])
        
        # Plot
        plt.plot(daily_avg['date'], daily_avg['response_time_seconds'], marker='o', linestyle='-', alpha=0.7)
        
        # Add vertical lines for period boundaries
        plt.axvline(x=baseline_start, color='green', linestyle='--', label='Baseline Start')
        plt.axvline(x=current_start, color='red', linestyle='--', label='Current Period Start')
        
        # Add horizontal lines for average values
        plt.axhline(y=result['baseline_value'], color='green', linestyle='-', alpha=0.5, label='Baseline Avg')
        plt.axhline(y=result['current_value'], color='red', linestyle='-', alpha=0.5, label='Current Avg')
        
        plt.title(f"Response Time Trend (Drift: {result['percent_change']:.2f}%)")
        plt.xlabel("Date")
        plt.ylabel("Response Time (seconds)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, f"response_time_drift_{datetime.now().strftime('%Y%m%d')}.png")
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Response time drift plot saved to {plot_path}")
    
    def _plot_feedback_trend(self, 
                            df: pd.DataFrame, 
                            baseline_start: datetime, 
                            current_start: datetime,
                            result: Dict[str, Any]) -> None:
        """
        Plot feedback quality trend.
        
        Args:
            df: DataFrame with feedback metrics
            baseline_start: Start of baseline period
            current_start: Start of current period
            result: Drift detection results
        """
        plt.figure(figsize=(12, 6))
        
        # Group by day and calculate average quality
        df['date'] = df['timestamp'].dt.date
        daily_avg = df.groupby('date')['alert_quality'].mean().reset_index()
        daily_avg['date'] = pd.to_datetime(daily_avg['date'])
        
        # Plot
        plt.plot(daily_avg['date'], daily_avg['alert_quality'], marker='o', linestyle='-', alpha=0.7)
        
        # Add vertical lines for period boundaries
        plt.axvline(x=baseline_start, color='green', linestyle='--', label='Baseline Start')
        plt.axvline(x=current_start, color='red', linestyle='--', label='Current Period Start')
        
        # Add horizontal lines for average values
        plt.axhline(y=result['baseline_value'], color='green', linestyle='-', alpha=0.5, label='Baseline Avg')
        plt.axhline(y=result['current_value'], color='red', linestyle='-', alpha=0.5, label='Current Avg')
        
        plt.title(f"Alert Quality Trend (Change: {result['percent_change']:.2f}%)")
        plt.xlabel("Date")
        plt.ylabel("Alert Quality Rating (1-5)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, f"alert_quality_drift_{datetime.now().strftime('%Y%m%d')}.png")
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Alert quality drift plot saved to {plot_path}")
    
    def _plot_embedding_drift(self, 
                             baseline_embeddings: np.ndarray, 
                             current_embeddings: np.ndarray,
                             result: Dict[str, Any]) -> None:
        """
        Plot embedding drift using PCA.
        
        Args:
            baseline_embeddings: Previous embeddings
            current_embeddings: Current embeddings
            result: Drift detection results
        """
        # Use PCA to reduce dimensionality to 2D
        pca = PCA(n_components=2)
        
        # Sample to limit plot size (max 1000 points from each set)
        max_samples = 1000
        baseline_sample = baseline_embeddings[:min(len(baseline_embeddings), max_samples)]
        current_sample = current_embeddings[:min(len(current_embeddings), max_samples)]
        
        # Combine for PCA fitting
        combined = np.vstack([baseline_sample, current_sample])
        pca_result = pca.fit_transform(combined)
        
        # Split back
        baseline_pca = pca_result[:len(baseline_sample)]
        current_pca = pca_result[len(baseline_sample):]
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(baseline_pca[:, 0], baseline_pca[:, 1], alpha=0.5, label='Baseline Embeddings', color='blue')
        plt.scatter(current_pca[:, 0], current_pca[:, 1], alpha=0.5, label='Current Embeddings', color='red')
        
        plt.title(f"Embedding Space Drift (Cosine Distance: {result['distance']:.4f})")
        plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, f"embedding_drift_{datetime.now().strftime('%Y%m%d')}.png")
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Embedding drift plot saved to {plot_path}")
    
    def save_embedding_sample(self, embeddings: np.ndarray, timestamp: Optional[datetime] = None) -> None:
        """
        Save a sample of embeddings for drift monitoring.
        
        Args:
            embeddings: Embedding vectors to save
            timestamp: Timestamp for the sample (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Convert timestamp to string
        timestamp_str = timestamp.isoformat()
        
        # Sample embeddings if too large (max 1000)
        max_samples = 1000
        if len(embeddings) > max_samples:
            indices = np.random.choice(len(embeddings), max_samples, replace=False)
            embeddings = embeddings[indices]
        
        # Save with timestamp
        np.savez(
            self.embedding_samples_file,
            embeddings=embeddings,
            timestamp=timestamp_str
        )
        
        logger.info(f"Saved {len(embeddings)} embedding samples at {timestamp_str}")
    
    def load_embedding_sample(self) -> Tuple[Optional[np.ndarray], Optional[datetime]]:
        """
        Load the most recent embedding sample.
        
        Returns:
            Tuple of (embeddings, timestamp)
        """
        if not os.path.exists(self.embedding_samples_file):
            logger.warning("No embedding samples file found")
            return None, None
        
        try:
            data = np.load(self.embedding_samples_file)
            embeddings = data['embeddings']
            timestamp = datetime.fromisoformat(str(data['timestamp']))
            return embeddings, timestamp
        
        except Exception as e:
            logger.error(f"Error loading embedding samples: {e}")
            return None, None
    
    def run_all_drift_checks(self) -> Dict[str, Any]:
        """
        Run all available drift checks.
        
        Returns:
            Dictionary with all drift check results
        """
        results = {}
        
        # Response time drift
        results['response_time'] = self.detect_response_time_drift()
        
        # Feedback drift
        results['feedback'] = self.detect_feedback_drift()
        
        # Embedding drift (if samples available)
        baseline_embeddings, baseline_timestamp = self.load_embedding_sample()
        if baseline_embeddings is not None and len(baseline_embeddings) > 0:
            # In a real system, you would compare with current embeddings
            # For now, we'll just simulate with a slightly modified version
            current_embeddings = baseline_embeddings + np.random.normal(0, 0.05, baseline_embeddings.shape)
            results['embeddings'] = self.detect_embedding_drift(
                baseline_embeddings, current_embeddings
            )
        
        # Calculate overall drift status
        any_drift = any(check.get('drift_detected', False) for check in results.values())
        results['any_drift_detected'] = any_drift
        
        return results

if __name__ == "__main__":
    # Example usage
    detector = DriftDetector()
    drift_results = detector.run_all_drift_checks()
    print(f"Drift detected: {drift_results['any_drift_detected']}")
    for check_name, result in drift_results.items():
        if check_name != 'any_drift_detected':
            status = result.get('drift_detected', False)
            print(f"{check_name}: {'DRIFT DETECTED' if status else 'OK'}") 