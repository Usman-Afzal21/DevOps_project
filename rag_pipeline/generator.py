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
import json

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

# Model constants
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your-api-key-here")
LLM_MODEL = "llama-3.3-70b-versatile"

# System prompt for RAG
SYSTEM_PROMPT = """You are an AI assistant for a banking system that generates detailed alerts based on transaction data.
Your task is to analyze the provided context and generate a JSON alert with the following components:
1. A clear, attention-grabbing title that summarizes the issue
2. A detailed description of what was detected and why it matters
3. A list of recommended actions that banking staff should take

Format your response as a valid JSON object with the following structure:
{
  "title": "Clear, specific alert title",
  "description": "Detailed description of the alert, including what was detected and why it matters",
  "recommended_actions": ["Action 1", "Action 2", "Action 3"]
}

Make your alerts professional, actionable, and valuable for banking staff.
"""

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

def get_groq_client(api_key=None):
    """
    Get or create a Groq client.
    
    Args:
        api_key: Optional API key for Groq
        
    Returns:
        Groq client instance
    """
    from groq import Groq
    
    # Use provided key, environment variable, or default from settings
    if api_key:
        client_api_key = api_key
    else:
        # Try to get from environment
        client_api_key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY)
    
    # Initialize and return client
    return Groq(api_key=client_api_key)

def generate_alert(query, timeframe=None, branch=None, severity=None, groq_api_key=None):
    """
    Generate an alert based on a query and optional parameters.
    
    Args:
        query: The query or description for alert generation
        timeframe: Optional timeframe for the alert
        branch: Optional branch ID for the alert
        severity: Optional severity for the alert
        groq_api_key: Optional API key for Groq
    
    Returns:
        Dictionary with alert details
    """
    logger.info(f"Generating general alert for query: {query}")
    
    # Try using the RAG pipeline first
    try:
        # Initialize vector store and retriever
        vector_store = TransactionVectorStore(persist_directory="data/vector_db")
        
        # Create collections safely - if they exist already, this won't fail
        try:
            vector_store.create_transaction_collection()
            vector_store.create_summary_collection()
        except Exception as collection_error:
            logger.warning(f"Error creating collections: {collection_error}. Continuing with existing collections.")
        
        # Initialize retriever with vector store
        retriever = TransactionRetriever(vector_store)
        
        # Get context from vector store with better error handling
        try:
            context = retriever.retrieve(query)
            # Format context for the prompt
            formatted_context = "\n\n".join([f"Source {i+1}:\n{doc}" for i, doc in enumerate(context)])
        except Exception as retrieval_error:
            logger.warning(f"Error retrieving context: {retrieval_error}. Proceeding without context.")
            formatted_context = "No relevant context found due to retrieval error."
        
        # Create message structure with context
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}\n\nContext:\n{formatted_context}"}
        ]
        
        # Include optional parameters if provided
        query_params = {}
        if timeframe:
            query_params["timeframe"] = timeframe
        if branch:
            query_params["branch"] = branch
        if severity:
            query_params["severity"] = severity
        
        if query_params:
            params_str = ", ".join([f"{k}={v}" for k, v in query_params.items()])
            messages.append({"role": "user", "content": f"Additional parameters: {params_str}"})
        
        # Use Groq LLM for generation with RAG context
        groq_response = get_groq_client(groq_api_key).chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        # Extract and parse the response
        content = groq_response.choices[0].message.content
        result = json.loads(content)
        
        logger.info(f"Successfully generated RAG response ({len(content)} chars)")
        return result
        
    except Exception as e:
        logger.error(f"RAG pipeline error: {str(e)}. Trying direct LLM approach.")
        
        # Fallback to direct LLM approach without RAG
        try:
            # Generate a system message for alert generation
            system_message = """You are a banking alert system that generates detailed, actionable alerts based on transaction data.
Generate a JSON response with the following structure:
{
  "title": "Clear, specific alert title",
  "description": "Detailed description of the alert, including what was detected and why it matters",
  "recommended_actions": ["Action 1", "Action 2", "Action 3"]
}
Make the alert specific, actionable, and valuable for banking professionals."""
            
            # Create a prompt for direct generation
            direct_prompt = f"""Generate a detailed banking alert based on this query: {query}
            
If timeframe is specified: {timeframe if timeframe else 'No specific timeframe'}
If branch is specified: {branch if branch else 'No specific branch'}
If severity is specified: {severity if severity else 'No specific severity'}

Provide a clear title, detailed description of what was detected, and 3-5 specific recommended actions.
"""
            
            # Use Groq LLM for direct generation
            direct_response = get_groq_client(groq_api_key).chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": direct_prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # Extract and try to parse the response
            content = direct_response.choices[0].message.content
            logger.info(f"Successfully generated direct response ({len(content)} chars)")
            
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, create a simple structured response
                logger.warning("Failed to parse LLM response as JSON. Creating basic structure.")
                
                # Extract title and description using basic string manipulation
                lines = content.strip().split('\n')
                title = next((line for line in lines if line.strip()), "Alert Generated")
                description = content
                
                return {
                    "title": title[:100],  # Limit title length
                    "description": description,
                    "recommended_actions": ["Review the alert details", "Investigate related transactions", "Consider updating monitoring rules"]
                }
                
        except Exception as inner_e:
            logger.error(f"Direct LLM approach also failed: {str(inner_e)}. Returning basic alert.")
            
            # Return a very basic alert if all else fails
            return {
                "title": f"Alert for: {query[:50]}...",
                "description": f"Alert generated for query: {query}",
                "recommended_actions": ["Review system logs", "Investigate alert generation failure", "Contact support if issue persists"]
            }

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

