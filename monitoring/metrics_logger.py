"""
Metrics Logging Module for MLOps Monitoring.

This module handles:
- Logging performance metrics
- Tracking LLM usage and costs
- Monitoring model drift
- Logging user feedback on alert quality
"""

import os
import json
import time
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import mlflow

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsLogger:
    """
    Track and store metrics for the RAG alert system.
    """
    
    def __init__(self, log_dir: str = "monitoring/logs", use_mlflow: bool = False):
        """
        Initialize the metrics logger.
        
        Args:
            log_dir: Directory to store logs
            use_mlflow: Whether to use MLflow for tracking
        """
        self.log_dir = log_dir
        self.use_mlflow = use_mlflow
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log files
        self.performance_log_file = os.path.join(log_dir, "performance_metrics.jsonl")
        self.usage_log_file = os.path.join(log_dir, "usage_metrics.jsonl")
        self.feedback_log_file = os.path.join(log_dir, "feedback_metrics.jsonl")
        
        # Initialize MLflow if enabled
        if use_mlflow:
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
            mlflow.set_experiment("rag-alert-system")
    
    def log_performance_metrics(self, 
                               query_id: str,
                               query_type: str,
                               response_time_seconds: float,
                               tokens_used: int,
                               embedding_time_seconds: Optional[float] = None,
                               retrieval_time_seconds: Optional[float] = None,
                               llm_time_seconds: Optional[float] = None) -> None:
        """
        Log performance metrics for an alert generation request.
        
        Args:
            query_id: Unique identifier for the query
            query_type: Type of alert generated
            response_time_seconds: Total response time in seconds
            tokens_used: Number of tokens used by the LLM
            embedding_time_seconds: Time spent on embeddings (optional)
            retrieval_time_seconds: Time spent on retrieval (optional)
            llm_time_seconds: Time spent on LLM generation (optional)
        """
        timestamp = datetime.now().isoformat()
        
        metrics = {
            "timestamp": timestamp,
            "query_id": query_id,
            "query_type": query_type,
            "response_time_seconds": response_time_seconds,
            "tokens_used": tokens_used,
            "embedding_time_seconds": embedding_time_seconds,
            "retrieval_time_seconds": retrieval_time_seconds,
            "llm_time_seconds": llm_time_seconds
        }
        
        # Log to file
        with open(self.performance_log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        
        # Log to MLflow
        if self.use_mlflow:
            with mlflow.start_run(run_name=f"query_{query_id}"):
                mlflow.log_metrics({
                    "response_time_seconds": response_time_seconds,
                    "tokens_used": tokens_used
                })
                
                if embedding_time_seconds:
                    mlflow.log_metric("embedding_time_seconds", embedding_time_seconds)
                if retrieval_time_seconds:
                    mlflow.log_metric("retrieval_time_seconds", retrieval_time_seconds)
                if llm_time_seconds:
                    mlflow.log_metric("llm_time_seconds", llm_time_seconds)
                
                mlflow.log_param("query_type", query_type)
        
        logger.info(f"Logged performance metrics for query {query_id}")
    
    def log_usage_metrics(self,
                         query_id: str,
                         model_name: str,
                         tokens_prompt: int,
                         tokens_completion: int,
                         estimated_cost_usd: float) -> None:
        """
        Log usage and cost metrics for LLM calls.
        
        Args:
            query_id: Unique identifier for the query
            model_name: Name of the LLM model used
            tokens_prompt: Number of tokens in the prompt
            tokens_completion: Number of tokens in the completion
            estimated_cost_usd: Estimated cost in USD
        """
        timestamp = datetime.now().isoformat()
        
        metrics = {
            "timestamp": timestamp,
            "query_id": query_id,
            "model_name": model_name,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "tokens_total": tokens_prompt + tokens_completion,
            "estimated_cost_usd": estimated_cost_usd
        }
        
        # Log to file
        with open(self.usage_log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        
        # Log to MLflow
        if self.use_mlflow:
            with mlflow.start_run(run_name=f"usage_{query_id}"):
                mlflow.log_metrics({
                    "tokens_prompt": tokens_prompt,
                    "tokens_completion": tokens_completion,
                    "tokens_total": tokens_prompt + tokens_completion,
                    "estimated_cost_usd": estimated_cost_usd
                })
                
                mlflow.log_param("model_name", model_name)
        
        logger.info(f"Logged usage metrics for query {query_id}")
    
    def log_feedback(self, 
                    query_id: str,
                    alert_quality: int,
                    is_accurate: bool,
                    is_actionable: bool,
                    user_comments: Optional[str] = None) -> None:
        """
        Log user feedback on the quality of alerts.
        
        Args:
            query_id: Unique identifier for the query
            alert_quality: Rating from 1-5 (5 being best)
            is_accurate: Whether the alert information was accurate
            is_actionable: Whether the alert was actionable
            user_comments: Optional user comments
        """
        timestamp = datetime.now().isoformat()
        
        metrics = {
            "timestamp": timestamp,
            "query_id": query_id,
            "alert_quality": alert_quality,
            "is_accurate": is_accurate,
            "is_actionable": is_actionable,
            "user_comments": user_comments
        }
        
        # Log to file
        with open(self.feedback_log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        
        # Log to MLflow
        if self.use_mlflow:
            with mlflow.start_run(run_name=f"feedback_{query_id}"):
                mlflow.log_metrics({
                    "alert_quality": alert_quality,
                    "is_accurate": int(is_accurate),
                    "is_actionable": int(is_actionable)
                })
                
                if user_comments:
                    mlflow.log_param("user_comments", user_comments)
        
        logger.info(f"Logged feedback for query {query_id}")
    
    def get_performance_summary(self, last_n_days: int = 7) -> pd.DataFrame:
        """
        Get a summary of performance metrics for the last N days.
        
        Args:
            last_n_days: Number of days to include in the summary
            
        Returns:
            DataFrame with performance metrics
        """
        if not os.path.exists(self.performance_log_file):
            logger.warning("Performance log file does not exist")
            return pd.DataFrame()
        
        # Read the log file
        try:
            df = pd.read_json(self.performance_log_file, lines=True)
            
            # Filter by date
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=last_n_days)
            df = df[df['timestamp'] >= cutoff_date]
            
            return df
        
        except Exception as e:
            logger.error(f"Error reading performance log file: {e}")
            return pd.DataFrame()
    
    def get_usage_summary(self, last_n_days: int = 7) -> pd.DataFrame:
        """
        Get a summary of usage metrics for the last N days.
        
        Args:
            last_n_days: Number of days to include in the summary
            
        Returns:
            DataFrame with usage metrics
        """
        if not os.path.exists(self.usage_log_file):
            logger.warning("Usage log file does not exist")
            return pd.DataFrame()
        
        # Read the log file
        try:
            df = pd.read_json(self.usage_log_file, lines=True)
            
            # Filter by date
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=last_n_days)
            df = df[df['timestamp'] >= cutoff_date]
            
            return df
        
        except Exception as e:
            logger.error(f"Error reading usage log file: {e}")
            return pd.DataFrame()
    
    def detect_performance_drift(self, 
                               baseline_period_days: int = 30, 
                               current_period_days: int = 7,
                               threshold_percent: float = 20.0) -> Dict[str, Any]:
        """
        Detect drift in performance metrics.
        
        Args:
            baseline_period_days: Number of days to use as baseline
            current_period_days: Number of days to compare against baseline
            threshold_percent: Threshold percentage for drift detection
            
        Returns:
            Dictionary with drift detection results
        """
        if not os.path.exists(self.performance_log_file):
            logger.warning("Performance log file does not exist")
            return {"drift_detected": False, "reason": "No data available"}
        
        try:
            df = pd.read_json(self.performance_log_file, lines=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Define baseline and current periods
            now = pd.Timestamp.now()
            baseline_start = now - pd.Timedelta(days=baseline_period_days)
            current_start = now - pd.Timedelta(days=current_period_days)
            
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
            
            return {
                "drift_detected": drift_detected,
                "metric": "response_time_seconds",
                "baseline_value": baseline_response_time,
                "current_value": current_response_time,
                "percent_change": response_time_change,
                "threshold": threshold_percent
            }
        
        except Exception as e:
            logger.error(f"Error detecting performance drift: {e}")
            return {"drift_detected": False, "reason": f"Error: {str(e)}"}

def performance_timer(func):
    """
    Decorator to time function execution and log performance metrics.
    
    Args:
        func: Function to time
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Get the first argument (self)
        if len(args) > 0 and hasattr(args[0], 'query_id'):
            self_obj = args[0]
            if hasattr(self_obj, 'metrics_logger') and self_obj.metrics_logger:
                self_obj.metrics_logger.log_performance_metrics(
                    query_id=self_obj.query_id,
                    query_type=func.__name__,
                    response_time_seconds=execution_time,
                    tokens_used=getattr(self_obj, 'tokens_used', 0)
                )
        
        return result
    
    return wrapper 