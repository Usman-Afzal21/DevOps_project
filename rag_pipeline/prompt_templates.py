"""
Prompt Templates Module for RAG-based Alert Generation.

This module defines the prompt templates for various alert generation scenarios.
"""

from typing import List, Dict, Any, Optional

# Template for general alert generation
ALERT_GENERATION_TEMPLATE = """
You are an AI assistant for a bank's Management Information System (MIS).
Your task is to analyze banking transaction data and provide meaningful alerts or insights.

Context information from the bank's transaction records:
{context}

Based on the above context, please generate a concise alert or insight for the following query:
Query: {query}

Your response should be clear, professional, and actionable. Focus on identifying patterns, anomalies, 
or trends that would be valuable for bank management to know about.
"""

# Template for branch-specific analysis
BRANCH_ANALYSIS_TEMPLATE = """
You are an AI assistant for a bank's Management Information System (MIS).
Your task is to analyze transaction data for a specific branch and provide insights.

Branch: {branch}

Context information from the branch's transaction records:
{context}

Time period: {time_period}

Based on the above context, please analyze the branch performance and identify any notable patterns, 
anomalies, or trends. Generate a concise alert or insight summary that would be valuable for bank management.

Your response should be clear, professional, and actionable.
"""

# Template for fraud risk assessment
FRAUD_RISK_TEMPLATE = """
You are an AI assistant for a bank's Fraud Management System.
Your task is to analyze transaction data and assess potential fraud risks.

Context information from recent transactions:
{context}

Based on the above context, please assess the fraud risk and generate a concise alert 
if you detect any suspicious patterns or anomalies that may indicate fraudulent activity.

Consider factors such as:
- Unusual transaction amounts
- Multiple failed login attempts
- Geographic inconsistencies
- Unusual timing or frequency of transactions
- Deviations from typical customer behavior

Your response should be clear, professional, and actionable for the fraud investigation team.
"""

# Template for comparative analysis between branches
COMPARATIVE_ANALYSIS_TEMPLATE = """
You are an AI assistant for a bank's Management Information System (MIS).
Your task is to compare performance metrics between branches and highlight significant differences.

Context information from Branch A:
{branch_a_context}

Context information from Branch B:
{branch_b_context}

Time period: {time_period}

Based on the above context, please compare the performance of Branch A and Branch B, 
highlighting significant differences, trends, or anomalies between them.

Your response should be clear, professional, and actionable for bank management.
"""

# Template for historical trend analysis
HISTORICAL_TREND_TEMPLATE = """
You are an AI assistant for a bank's Management Information System (MIS).
Your task is to analyze historical transaction data and identify meaningful trends over time.

Context information from the bank's transaction records over time:
{context}

Time period: {time_period}

Based on the above context, please analyze the historical trends in the data and identify 
any patterns, seasonal variations, or significant changes over time.

Your response should be clear, professional, and actionable for bank management.
"""

def get_alert_prompt(query: str, context: str) -> str:
    """
    Get a formatted alert generation prompt.
    
    Args:
        query: The user's query about transaction data
        context: The retrieved transaction context information
    
    Returns:
        Formatted prompt string
    """
    return ALERT_GENERATION_TEMPLATE.format(
        query=query,
        context=context
    )

def get_branch_analysis_prompt(branch: str, context: str, time_period: str = "recent") -> str:
    """
    Get a formatted branch analysis prompt.
    
    Args:
        branch: The branch name to analyze
        context: The retrieved branch context information
        time_period: Description of the time period for analysis
    
    Returns:
        Formatted prompt string
    """
    return BRANCH_ANALYSIS_TEMPLATE.format(
        branch=branch,
        context=context,
        time_period=time_period
    )

def get_fraud_risk_prompt(context: str) -> str:
    """
    Get a formatted fraud risk assessment prompt.
    
    Args:
        context: The retrieved transaction context information
    
    Returns:
        Formatted prompt string
    """
    return FRAUD_RISK_TEMPLATE.format(
        context=context
    )

def get_comparative_prompt(branch_a_context: str, branch_b_context: str, time_period: str = "recent") -> str:
    """
    Get a formatted comparative analysis prompt.
    
    Args:
        branch_a_context: The retrieved context for Branch A
        branch_b_context: The retrieved context for Branch B
        time_period: Description of the time period for comparison
    
    Returns:
        Formatted prompt string
    """
    return COMPARATIVE_ANALYSIS_TEMPLATE.format(
        branch_a_context=branch_a_context,
        branch_b_context=branch_b_context,
        time_period=time_period
    )

def get_trend_analysis_prompt(context: str, time_period: str) -> str:
    """
    Get a formatted historical trend analysis prompt.
    
    Args:
        context: The retrieved transaction context information
        time_period: Description of the time period for analysis
    
    Returns:
        Formatted prompt string
    """
    return HISTORICAL_TREND_TEMPLATE.format(
        context=context,
        time_period=time_period
    ) 