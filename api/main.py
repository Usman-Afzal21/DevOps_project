"""
FastAPI Server for AI-powered Alert Management System.

This module provides REST API endpoints for generating alerts and insights
from banking transaction data using RAG.
"""

import os
import sys
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add project directories to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from rag_pipeline.generator import generate_alert, test_groq_client

# No need to load environment variables

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI-powered Alert Management System",
    description="Generate intelligent alerts from banking transaction data using RAG.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class AlertRequest(BaseModel):
    query: str
    branch: Optional[str] = None
    alert_type: str = "general"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    compare_with: Optional[str] = None
    timeframe: str = "recent"

class AlertResponse(BaseModel):
    alert: str
    alert_type: str
    query: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Define API routes
@app.get("/")
async def root():
    """Root endpoint for the API."""
    return {"message": "Welcome to the AI-powered Alert Management System API"}

@app.post("/generate-alert", response_model=AlertResponse)
async def api_generate_alert(request: AlertRequest):
    """
    Generate an alert based on the request parameters.
    
    - query: User query or topic for the alert
    - branch: Branch to focus on (if applicable)
    - alert_type: Type of alert to generate (general, branch, fraud, comparison, trend)
    - start_date: Start date for time range (if applicable, format: YYYY-MM-DD)
    - end_date: End date for time range (if applicable, format: YYYY-MM-DD)
    - compare_with: Second branch for comparison (if applicable)
    - timeframe: Description of the time period (e.g., "recent", "last month")
    """
    try:
        logger.info(f"Received alert request: {request.dict()}")
        
        # Generate alert
        alert = generate_alert(
            query=request.query,
            branch=request.branch,
            start_date=request.start_date,
            end_date=request.end_date,
            compare_with=request.compare_with,
            timeframe=request.timeframe
        )
        
        # Prepare metadata
        metadata = {
            "branch": request.branch,
            "timeframe": request.timeframe
        }
        
        if request.start_date:
            metadata["start_date"] = request.start_date
        if request.end_date:
            metadata["end_date"] = request.end_date
        if request.compare_with:
            metadata["compare_with"] = request.compare_with
        
        return AlertResponse(
            alert=alert,
            alert_type=request.alert_type,
            query=request.query,
            metadata=metadata
        )
    
    except Exception as e:
        logger.error(f"Error generating alert: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating alert: {str(e)}")

@app.get("/branch-list")
async def get_branch_list():
    """
    Get a list of available bank branches. This is a mock endpoint for the UI.
    """
    # This is a mock list based on the data we've seen
    return {
        "branches": [
            "New York", "San Francisco", "Chicago", "Houston", "Los Angeles",
            "Miami", "Seattle", "Boston", "Dallas", "Denver", "Atlanta", 
            "Phoenix", "Philadelphia", "San Diego", "Washington", "Detroit",
            "Las Vegas", "Portland", "San Antonio", "Nashville"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}

# Add specific alert endpoints
@app.get("/alerts/general")
async def get_general_alerts():
    """Get general alerts about banking operations."""
    try:
        alert = generate_alert(
            query="Generate a general overview alert about banking operations",
            timeframe="recent"
        )
        return {"alert": alert, "alert_type": "general"}
    except Exception as e:
        logger.error(f"Error generating general alert: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating alert: {str(e)}")

@app.get("/alerts/branch/{branch_id}")
async def get_branch_alerts(branch_id: str):
    """Get alerts for a specific branch."""
    try:
        alert = generate_alert(
            query=f"Generate insights for {branch_id} branch",
            branch=branch_id,
            timeframe="recent"
        )
        return {"alert": alert, "alert_type": "branch", "branch": branch_id}
    except Exception as e:
        logger.error(f"Error generating branch alert: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating alert: {str(e)}")

@app.get("/alerts/fraud")
async def get_fraud_alerts():
    """Get fraud detection alerts."""
    try:
        alert = generate_alert(
            query="Detect potential fraud patterns in recent transactions",
            timeframe="recent"
        )
        return {"alert": alert, "alert_type": "fraud"}
    except Exception as e:
        logger.error(f"Error generating fraud alert: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating alert: {str(e)}")

@app.get("/alerts/comparison")
async def get_comparison_alerts(
    branch1: str = Query(..., description="First branch for comparison"),
    branch2: str = Query(..., description="Second branch for comparison")
):
    """Get alerts comparing two branches."""
    try:
        alert = generate_alert(
            query=f"Compare performance between {branch1} and {branch2} branches",
            branch=branch1,
            compare_with=branch2,
            timeframe="recent"
        )
        return {"alert": alert, "alert_type": "comparison", "branch1": branch1, "branch2": branch2}
    except Exception as e:
        logger.error(f"Error generating comparison alert: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating alert: {str(e)}")

@app.get("/alerts/trends")
async def get_trend_alerts(
    branch: str = Query(..., description="Branch to analyze"),
    period: str = Query("month", description="Time period (day, week, month, quarter, year)")
):
    """Get trend analysis alerts for a branch over a time period."""
    try:
        alert = generate_alert(
            query=f"Analyze trends for {branch} branch over the last {period}",
            branch=branch,
            timeframe=period
        )
        return {"alert": alert, "alert_type": "trend", "branch": branch, "period": period}
    except Exception as e:
        logger.error(f"Error generating trend alert: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating alert: {str(e)}")

@app.get("/test-groq")
async def test_groq():
    """A simple test endpoint to directly test the Groq API without RAG."""
    try:
        response = test_groq_client()
        return {"response": response}
    except Exception as e:
        logger.error(f"Error testing Groq client: {e}")
        raise HTTPException(status_code=500, detail=f"Error testing Groq client: {str(e)}")

if __name__ == "__main__":
    # Run the API server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

