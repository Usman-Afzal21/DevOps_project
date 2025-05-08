"""
Alert Generator Module for RAG Pipeline.

This module handles:
- Integrating with the Groq API for LLM access
- Generating alerts based on context and prompts
- Managing different types of alert generation
"""

import os
import logging
from groq import Groq  # Import Groq client
from typing import List, Dict, Any, Optional, Tuple
import sys
import time

# No dotenv loading - direct API key usage

# Add project directories to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rag_pipeline.prompt_templates import (
    get_alert_prompt, get_branch_analysis_prompt, 
    get_fraud_risk_prompt, get_comparative_prompt,
    get_trend_analysis_prompt
)
from rag_pipeline.retriever import TransactionRetriever
from embeddings.vector_store import TransactionVectorStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hardcoded API key for Groq
GROQ_API_KEY = "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"

class AlertGenerator:
    """
    Generate AI-powered alerts using LLMs with RAG functionality.
    """
    
    def __init__(self, retriever: TransactionRetriever, model_name: str = "llama-3.3-70b-versatile"):
        """
        Initialize the alert generator.
        
        Args:
            retriever: A TransactionRetriever instance
            model_name: Groq model name to use
        """
        self.retriever = retriever
        self.model_name = model_name
        
        # Initialize Groq client correctly according to the latest documentation
        self.client = Groq(api_key=GROQ_API_KEY)
        logger.info(f"Initialized Groq client with model: {model_name}")
    
    def generate_llm_response(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Generate a response from the LLM using Groq API.
        
        Args:
            prompt: The formatted prompt to send to the LLM
            max_tokens: Maximum tokens to generate in response
            
        Returns:
            Generated response string
        """
        try:
            logger.info(f"Generating response using {self.model_name}")
            
            # Debug client info
            logger.info(f"Groq client: {self.client}")
            logger.info(f"Groq client type: {type(self.client)}")
            logger.info(f"Groq client dir: {dir(self.client)}")
            
            # Add retry mechanism for API stability
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Updated to follow the latest Groq API documentation
                    # More debug info
                    logger.info("Attempting to call chat.completions.create")
                    logger.info(f"chat: {self.client.chat}")
                    logger.info(f"completions: {self.client.chat.completions}")
                    
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a banking AI assistant that generates concise, professional alerts based on transaction data."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=max_tokens,
                        temperature=0.2  # Lower temperature for more focused outputs
                    )
                    
                    response = completion.choices[0].message.content
                    logger.info(f"Successfully generated response ({len(response)} chars)")
                    return response
                
                except Exception as e:
                    retry_count += 1
                    logger.error(f"API error details: {type(e).__name__}: {str(e)}")
                    if retry_count < max_retries:
                        logger.warning(f"API error: {e}. Retrying {retry_count}/{max_retries}...")
                        time.sleep(2)  # Wait before retrying
                    else:
                        raise
        
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error generating alert: {str(e)}"
    
    def generate_general_alert(self, query: str) -> str:
        """
        Generate a general alert based on a user query.
        
        Args:
            query: The user's query about transaction data
            
        Returns:
            Generated alert string
        """
        logger.info(f"Generating general alert for query: {query}")
        
        # Retrieve relevant context
        context = self.retriever.retrieve_for_query(query)
        
        # Generate prompt
        prompt = get_alert_prompt(query, context)
        
        # Generate response
        return self.generate_llm_response(prompt)
    
    def generate_branch_alert(self, branch: str, time_period: str = "recent") -> str:
        """
        Generate an alert for a specific branch.
        
        Args:
            branch: The branch to analyze
            time_period: Description of the time period
            
        Returns:
            Generated alert string
        """
        logger.info(f"Generating branch alert for {branch}")
        
        # Retrieve branch-specific context
        context = self.retriever.retrieve_for_branch(branch)
        
        # Generate prompt
        prompt = get_branch_analysis_prompt(branch, context, time_period)
        
        # Generate response
        return self.generate_llm_response(prompt)
    
    def generate_fraud_alert(self, query: str = "potential fraud patterns") -> str:
        """
        Generate a fraud risk assessment alert.
        
        Args:
            query: Specific fraud query (optional)
            
        Returns:
            Generated alert string
        """
        logger.info(f"Generating fraud alert for: {query}")
        
        # Retrieve fraud-relevant context
        context = self.retriever.retrieve_for_query(query)
        
        # Generate prompt
        prompt = get_fraud_risk_prompt(context)
        
        # Generate response
        return self.generate_llm_response(prompt)
    
    def generate_comparative_alert(self, branch_a: str, branch_b: str, time_period: str = "recent") -> str:
        """
        Generate a comparative analysis between two branches.
        
        Args:
            branch_a: First branch to compare
            branch_b: Second branch to compare
            time_period: Description of the time period
            
        Returns:
            Generated comparative alert string
        """
        logger.info(f"Generating comparative alert for {branch_a} vs {branch_b}")
        
        # Retrieve branch comparison contexts
        contexts = self.retriever.retrieve_for_comparison(branch_a, branch_b)
        
        # Generate prompt
        prompt = get_comparative_prompt(
            contexts['branch_a_context'], 
            contexts['branch_b_context'],
            time_period
        )
        
        # Generate response
        return self.generate_llm_response(prompt)
    
    def generate_trend_alert(self, start_date: str, end_date: str, branch: Optional[str] = None) -> str:
        """
        Generate a trend analysis alert for a time period.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            branch: Optional branch to focus on
            
        Returns:
            Generated trend alert string
        """
        logger.info(f"Generating trend alert for period {start_date} to {end_date}")
        
        # Retrieve date range context
        context = self.retriever.retrieve_for_date_range(start_date, end_date, branch)
        
        # Format time period string
        time_period = f"{start_date} to {end_date}"
        if branch:
            time_period += f" for {branch}"
        
        # Generate prompt
        prompt = get_trend_analysis_prompt(context, time_period)
        
        # Generate response
        return self.generate_llm_response(prompt)

def generate_alert(query: str, 
                  branch: Optional[str] = None, 
                  alert_type: str = "general",
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  compare_with: Optional[str] = None,
                  timeframe: str = "recent") -> str:
    """
    High-level function to generate alerts of various types.
    
    Args:
        query: User query or topic for the alert
        branch: Branch to focus on (if applicable)
        alert_type: Type of alert to generate ('general', 'branch', 'fraud', 'comparison', 'trend')
        start_date: Start date for time range (if applicable)
        end_date: End date for time range (if applicable)
        compare_with: Second branch for comparison (if applicable)
        timeframe: Description of the time period
        
    Returns:
        Generated alert string
    """
    try:
        # Try the RAG pipeline first
        try:
            # Initialize the vector store and retriever
            vector_store = TransactionVectorStore(persist_directory="data/vector_db")
            vector_store.create_transaction_collection()
            vector_store.create_summary_collection()
            
            retriever = TransactionRetriever(vector_store)
            
            # Create an AlertGenerator with direct Groq initialization
            generator = AlertGenerator(retriever, model_name="llama-3.3-70b-versatile")
            
            # Generate the appropriate type of alert
            if alert_type == "branch" and branch:
                return generator.generate_branch_alert(branch, timeframe)
            elif alert_type == "fraud":
                return generator.generate_fraud_alert(query)
            elif alert_type == "comparison" and branch and compare_with:
                return generator.generate_comparative_alert(branch, compare_with, timeframe)
            elif alert_type == "trend" and start_date and end_date:
                return generator.generate_trend_alert(start_date, end_date, branch)
            else:
                # Default to general alert
                return generator.generate_general_alert(query)
        except Exception as rag_error:
            # If RAG pipeline fails, log the error and try the direct approach
            logger.error(f"RAG pipeline error: {rag_error}. Trying direct LLM approach.")
            
            # Create a description based on the alert type
            if alert_type == "general":
                description = "a general overview of banking operations"
            elif alert_type == "branch" and branch:
                description = f"an analysis of the {branch} branch's performance during {timeframe}"
            elif alert_type == "fraud":
                description = "potential fraud patterns and risk assessment in banking transactions"
            elif alert_type == "comparison" and branch and compare_with:
                description = f"a comparison between {branch} and {compare_with} branches during {timeframe}"
            elif alert_type == "trend" and start_date and end_date:
                time_desc = f"from {start_date} to {end_date}"
                if branch:
                    time_desc += f" for {branch} branch"
                description = f"trend analysis of banking transactions {time_desc}"
            else:
                description = "insights about banking transactions based on available data"
            
            # Create a direct prompt
            prompt = f"""
            Generate {description}. 
            
            Query: {query}
            
            Your response should be professional, concise, and formatted as a banking alert
            with relevant insights, patterns, and recommendations.
            """
            
            # Use direct Groq client
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a banking AI assistant that generates concise, professional alerts based on transaction data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.2
            )
            
            response = completion.choices[0].message.content
            logger.info(f"Successfully generated direct response ({len(response)} chars)")
            return response
            
    except Exception as e:
        logger.error(f"Error generating alert: {str(e)}")
        return f"Error generating alert: {str(e)}"

def test_groq_client() -> str:
    """
    Test function that uses the Groq API directly without embeddings or retrieval.
    This bypasses the ChromaDB and other components that might be causing issues.
    
    Returns:
        Generated text response
    """
    try:
        # Initialize Groq client directly
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        
        logger.info("Testing direct Groq client call")
        
        # Simple completion request
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a banking AI assistant."},
                {"role": "user", "content": "Give me a brief overview of recent banking transaction trends."}
            ],
            max_tokens=1024,
            temperature=0.2
        )
        
        response = completion.choices[0].message.content
        logger.info(f"Successfully generated test response ({len(response)} chars)")
        return response
    
    except Exception as e:
        logger.error(f"Error in test_groq_client: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Error testing Groq client: {str(e)}"

if __name__ == "__main__":
    # Example usage
    try:
        # Initialize components
        vector_store = TransactionVectorStore(persist_directory="../data/vector_db")
        vector_store.create_transaction_collection()
        vector_store.create_summary_collection()
        
        retriever = TransactionRetriever(vector_store)
        generator = AlertGenerator(retriever, model_name="llama-3.3-70b-versatile")
        
        # Generate a sample alert
        alert = generator.generate_general_alert("What are the most unusual transactions in the last month?")
        print(f"Generated Alert:\n{alert}")
    
    except Exception as e:
        logger.error(f"Error in example: {e}")
        print(f"Error: {str(e)}")

