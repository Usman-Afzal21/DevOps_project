import streamlit as st
import requests
import json
import pandas as pd
import datetime
import time

# Set page configuration
st.set_page_config(
    page_title="AI-powered Alert Management System",
    page_icon="🔔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_URL = "http://localhost:8000"
BRANCH_LIST = [
    "New York", "San Francisco", "Chicago", "Houston", "Los Angeles",
    "Miami", "Seattle", "Boston", "Dallas", "Denver", "Atlanta", 
    "Phoenix", "Philadelphia", "San Diego", "Washington", "Detroit",
    "Las Vegas", "Portland", "San Antonio", "Nashville"
]

# Functions to interact with the API
def get_general_alert():
    try:
        response = requests.get(f"{API_URL}/alerts/general")
        return response.json()["alert"]
    except Exception as e:
        st.error(f"Error fetching general alert: {str(e)}")
        return "Unable to fetch alert at this time."

def get_branch_alert(branch):
    try:
        response = requests.get(f"{API_URL}/alerts/branch/{branch}")
        return response.json()["alert"]
    except Exception as e:
        st.error(f"Error fetching branch alert: {str(e)}")
        return "Unable to fetch branch alert at this time."

def get_fraud_alert():
    try:
        response = requests.get(f"{API_URL}/alerts/fraud")
        return response.json()["alert"]
    except Exception as e:
        st.error(f"Error fetching fraud alert: {str(e)}")
        return "Unable to fetch fraud alert at this time."

def get_comparison_alert(branch1, branch2):
    try:
        response = requests.get(f"{API_URL}/alerts/comparison?branch1={branch1}&branch2={branch2}")
        return response.json()["alert"]
    except Exception as e:
        st.error(f"Error fetching comparison alert: {str(e)}")
        return "Unable to fetch comparison alert at this time."

def get_trend_alert(branch, period="month"):
    try:
        response = requests.get(f"{API_URL}/alerts/trends?branch={branch}&period={period}")
        return response.json()["alert"]
    except Exception as e:
        st.error(f"Error fetching trend alert: {str(e)}")
        return "Unable to fetch trend alert at this time."

def generate_custom_alert(query, alert_type, branch=None, compare_with=None, timeframe="recent"):
    try:
        payload = {
            "query": query,
            "alert_type": alert_type,
            "timeframe": timeframe
        }
        
        if branch:
            payload["branch"] = branch
        
        if compare_with:
            payload["compare_with"] = compare_with
            
        response = requests.post(f"{API_URL}/generate-alert", json=payload)
        return response.json()["alert"]
    except Exception as e:
        st.error(f"Error generating custom alert: {str(e)}")
        return "Unable to generate custom alert at this time."

# UI Components
def display_sidebar():
    st.sidebar.title("Navigation")
    pages = ["Dashboard", "Branch Analysis", "Fraud Detection", "Comparative Analysis", "Trend Analysis", "Custom Alert"]
    selected_page = st.sidebar.radio("Go to", pages)
    
    st.sidebar.markdown("---")
    st.sidebar.title("About")
    st.sidebar.info(
        "This is an AI-powered Alert Management System for banking data. "
        "It uses Retrieval-Augmented Generation (RAG) with Llama 3 via Groq API "
        "to generate intelligent alerts and insights."
    )
    
    return selected_page

def dashboard_page():
    st.title("Dashboard: AI-powered Alert Management System")
    st.subheader("Real-time Banking Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("General Banking Insights")
        with st.spinner("Generating general insights..."):
            alert = get_general_alert()
            st.write(alert)
    
    with col2:
        st.info("Fraud Detection Summary")
        with st.spinner("Analyzing fraud patterns..."):
            alert = get_fraud_alert()
            st.write(alert)
    
    st.markdown("---")
    
    st.subheader("Quick Branch Overview")
    selected_branch = st.selectbox("Select Branch", BRANCH_LIST)
    
    if st.button("Generate Branch Overview"):
        with st.spinner(f"Generating insights for {selected_branch}..."):
            alert = get_branch_alert(selected_branch)
            st.info(f"Branch Analysis: {selected_branch}")
            st.write(alert)

def branch_analysis_page():
    st.title("Branch Analysis")
    
    selected_branch = st.selectbox("Select Branch", BRANCH_LIST)
    timeframe_options = ["Today", "This Week", "This Month", "This Quarter", "This Year"]
    selected_timeframe = st.selectbox("Select Timeframe", timeframe_options)
    
    if st.button("Generate Branch Analysis"):
        with st.spinner(f"Analyzing {selected_branch} branch data..."):
            alert = get_branch_alert(selected_branch)
            
            st.subheader(f"Analysis for {selected_branch} - {selected_timeframe}")
            st.write(alert)
            
            # Display a mock visualization
            if alert != "Unable to fetch branch alert at this time.":
                st.subheader("Transaction Volume Trend")
                chart_data = pd.DataFrame({
                    'date': pd.date_range(start='2023-01-01', periods=30),
                    'volume': [100 + i + i**1.5 for i in range(30)]
                })
                st.line_chart(chart_data.set_index('date'))

def fraud_detection_page():
    st.title("Fraud Detection")
    
    alert_trigger = st.radio(
        "Alert Trigger",
        ["Automatic (AI-detected)", "Manual Investigation"]
    )
    
    if alert_trigger == "Automatic (AI-detected)":
        if st.button("Generate Fraud Risk Assessment"):
            with st.spinner("Analyzing transaction patterns for fraud..."):
                alert = get_fraud_alert()
                
                st.subheader("Fraud Risk Assessment")
                st.write(alert)
                
                # Display mock fraud risk score
                risk_score = 65  # This would be generated from the AI analysis
                st.subheader("Fraud Risk Score")
                st.progress(risk_score/100)
                st.caption(f"Risk Score: {risk_score}/100")
    else:
        st.subheader("Manual Fraud Investigation")
        transaction_id = st.text_input("Transaction ID to Investigate")
        
        if st.button("Investigate Transaction"):
            with st.spinner("Investigating transaction..."):
                # This would connect to the actual fraud detection endpoint
                st.info("Transaction Investigation Results")
                st.write("This transaction shows several unusual patterns consistent with potential fraud:")
                st.write("- Transaction amount significantly deviates from customer's normal behavior")
                st.write("- Geographic location differs from usual transaction locations")
                st.write("- Multiple transactions in short succession")

def comparative_analysis_page():
    st.title("Comparative Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        branch1 = st.selectbox("First Branch", BRANCH_LIST, index=0)
    
    with col2:
        filtered_branches = [b for b in BRANCH_LIST if b != branch1]
        branch2 = st.selectbox("Second Branch", filtered_branches, index=0)
    
    timeframe_options = ["This Week", "This Month", "This Quarter", "This Year"]
    selected_timeframe = st.selectbox("Timeframe", timeframe_options)
    
    if st.button("Generate Comparison"):
        with st.spinner(f"Comparing {branch1} and {branch2}..."):
            alert = get_comparison_alert(branch1, branch2)
            
            st.subheader(f"Comparison: {branch1} vs {branch2}")
            st.write(alert)
            
            # Display mock comparative visualization
            if alert != "Unable to fetch comparison alert at this time.":
                st.subheader("Transaction Volume Comparison")
                chart_data = pd.DataFrame({
                    'date': pd.date_range(start='2023-01-01', periods=10),
                    branch1: [100 + i*10 for i in range(10)],
                    branch2: [80 + i*12 for i in range(10)]
                })
                st.line_chart(chart_data.set_index('date'))

def trend_analysis_page():
    st.title("Trend Analysis")
    
    selected_branch = st.selectbox("Select Branch", BRANCH_LIST)
    
    period_options = ["day", "week", "month", "quarter", "year"]
    selected_period = st.selectbox("Analysis Period", period_options, index=2)
    
    if st.button("Generate Trend Analysis"):
        with st.spinner(f"Analyzing trends for {selected_branch}..."):
            alert = get_trend_alert(selected_branch, selected_period)
            
            st.subheader(f"Trend Analysis for {selected_branch} - Last {selected_period}")
            st.write(alert)
            
            # Display mock trend visualization
            if alert != "Unable to fetch trend alert at this time.":
                st.subheader("Transaction Volume Over Time")
                
                # Generate sample data based on period
                if selected_period == "day":
                    dates = pd.date_range(start=datetime.datetime.now() - datetime.timedelta(hours=24), 
                                         periods=24, freq='H')
                    data = [50 + i + i**1.2 for i in range(24)]
                elif selected_period == "week":
                    dates = pd.date_range(start=datetime.datetime.now() - datetime.timedelta(days=7), 
                                         periods=7, freq='D')
                    data = [100 + i*15 for i in range(7)]
                elif selected_period == "month":
                    dates = pd.date_range(start=datetime.datetime.now() - datetime.timedelta(days=30), 
                                         periods=30, freq='D')
                    data = [100 + i*5 + i**1.5 for i in range(30)]
                elif selected_period == "quarter":
                    dates = pd.date_range(start=datetime.datetime.now() - datetime.timedelta(days=90), 
                                         periods=12, freq='W')
                    data = [500 + i*50 for i in range(12)]
                else:  # year
                    dates = pd.date_range(start=datetime.datetime.now() - datetime.timedelta(days=365), 
                                         periods=12, freq='M')
                    data = [1000 + i*100 for i in range(12)]
                
                chart_data = pd.DataFrame({
                    'date': dates,
                    'volume': data
                })
                st.line_chart(chart_data.set_index('date'))

def custom_alert_page():
    st.title("Custom Alert Generator")
    
    alert_type = st.selectbox(
        "Alert Type",
        ["general", "branch", "fraud", "comparison", "trend"]
    )
    
    query = st.text_area("Custom Query", height=100, 
                         value="Analyze transaction patterns and provide insights")
    
    # Show relevant parameters based on alert type
    branch = None
    compare_with = None
    timeframe = "recent"
    
    if alert_type in ["branch", "comparison", "trend"]:
        branch = st.selectbox("Branch", BRANCH_LIST)
    
    if alert_type == "comparison":
        filtered_branches = [b for b in BRANCH_LIST if b != branch]
        compare_with = st.selectbox("Compare With", filtered_branches, index=0)
    
    if alert_type in ["branch", "comparison", "trend"]:
        timeframe_options = ["Today", "This Week", "This Month", "This Quarter", "This Year"]
        timeframe = st.selectbox("Timeframe", timeframe_options, index=2)
    
    if st.button("Generate Alert"):
        with st.spinner("Generating custom alert..."):
            alert = generate_custom_alert(query, alert_type, branch, compare_with, timeframe)
            
            st.subheader("Generated Alert")
            st.write(alert)

# Main app
def main():
    # Apply custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Display header
    st.markdown('<h1 class="main-header">🏦 Bank MIS Alert Management System</h1>', unsafe_allow_html=True)
    
    # Display sidebar and get selected page
    selected_page = display_sidebar()
    
    # Display the selected page
    if selected_page == "Dashboard":
        dashboard_page()
    elif selected_page == "Branch Analysis":
        branch_analysis_page()
    elif selected_page == "Fraud Detection":
        fraud_detection_page()
    elif selected_page == "Comparative Analysis":
        comparative_analysis_page()
    elif selected_page == "Trend Analysis":
        trend_analysis_page()
    elif selected_page == "Custom Alert":
        custom_alert_page()

if __name__ == "__main__":
    main() 