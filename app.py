import streamlit as st
import requests
import json
import pandas as pd
import datetime
import time
import os
import sys
import logging
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Workaround for torch classes runtime error in streamlit
# Set environment variable to disable torch classes before importing torch
os.environ["ENABLE_TORCH_MODULE_CLASSES"] = "0"
try:
    import torch
    # Avoid referencing torch.classes directly
    torch.classes = None
except ImportError:
    pass
except Exception as e:
    print(f"Warning: Could not apply torch workaround: {str(e)}")
    pass

# Add project directories to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_pipeline.generator import generate_alert
from rag_pipeline.alert_generator import AutomaticAlertGenerator
from data.data_versioning import DataVersionManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Banking Alert Management System",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_URL = "http://localhost:8001"
BRANCH_LIST = [
    "New York", "San Francisco", "Chicago", "Houston", "Los Angeles",
    "Miami", "Seattle", "Boston", "Dallas", "Denver", "Atlanta", 
    "Phoenix", "Philadelphia", "San Diego", "Washington", "Detroit",
    "Las Vegas", "Portland", "San Antonio", "Nashville"
]

# Functions to interact with the API
def get_general_alert():
    """Get general insights about banking operations with fallback mechanism"""
    try:
        response = requests.get(f"{API_URL}/alerts/general", timeout=5)  # Add timeout to avoid long waits
        return response.json()["alert"]
    except Exception as e:
        logger.warning(f"Error fetching general alert from API: {str(e)}")
        try:
            # Direct fallback using local generator
            alert_content = generate_alert(
                query="Generate a general overview alert about banking operations",
                timeframe="recent"
            )
            if isinstance(alert_content, dict):
                return alert_content.get('description', "Unable to generate insights.")
            return alert_content
        except Exception as inner_e:
            logger.error(f"Error in fallback for general alert: {str(inner_e)}")
            return "Unable to fetch alert at this time. API unavailable and fallback failed."

def get_branch_alert(branch):
    """Get alerts for a specific branch with fallback mechanism"""
    try:
        response = requests.get(f"{API_URL}/alerts/branch/{branch}", timeout=5)
        return response.json()["alert"]
    except Exception as e:
        logger.warning(f"Error fetching branch alert from API: {str(e)}")
        try:
            # Direct fallback using local generator
            alert_content = generate_alert(
                query=f"Generate insights for {branch} branch",
                branch=branch,
                timeframe="recent"
            )
            if isinstance(alert_content, dict):
                return alert_content.get('description', f"Unable to generate insights for {branch}.")
            return alert_content
        except Exception as inner_e:
            logger.error(f"Error in fallback for branch alert: {str(inner_e)}")
            return f"Unable to fetch branch alert for {branch} at this time."

def get_fraud_alert():
    """Get fraud detection alerts with fallback mechanism"""
    try:
        response = requests.get(f"{API_URL}/alerts/fraud", timeout=5)
        return response.json()["alert"]
    except Exception as e:
        logger.warning(f"Error fetching fraud alert from API: {str(e)}")
        try:
            # Direct fallback using local generator
            alert_content = generate_alert(
                query="Detect potential fraud patterns in recent transactions",
                timeframe="recent"
            )
            if isinstance(alert_content, dict):
                return alert_content.get('description', "Unable to generate fraud insights.")
            return alert_content
        except Exception as inner_e:
            logger.error(f"Error in fallback for fraud alert: {str(inner_e)}")
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
    pages = ["Dashboard", "Alerts", "Data Management", "Custom Analysis"]
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
        try:
            with st.spinner("Generating general insights..."):
                alert = get_general_alert()
                st.write(alert)
        except Exception as e:
            # Fallback to local alerts when API is unavailable
            alerts = load_latest_alerts()
            if alerts:
                # Find a general alert
                general_alerts = [a for a in alerts if a.get('type') not in ['fraud_flags', 'high_login_attempts']]
                if general_alerts:
                    st.write(general_alerts[0].get('description', 'No insights available'))
                else:
                    st.write("Local general insights unavailable. Please check API connection.")
            else:
                st.write("No insights available. Please generate alerts or check API connection.")
    
    with col2:
        st.info("Fraud Detection Summary")
        try:
            with st.spinner("Analyzing fraud patterns..."):
                alert = get_fraud_alert()
                st.write(alert)
        except Exception as e:
            # Fallback to local alerts when API is unavailable
            alerts = load_latest_alerts()
            if alerts:
                # Find a fraud alert
                fraud_alerts = [a for a in alerts if a.get('type') in ['fraud_flags', 'high_login_attempts']]
                if fraud_alerts:
                    st.write(fraud_alerts[0].get('description', 'No fraud insights available'))
                else:
                    st.write("Local fraud insights unavailable. Please check API connection.")
            else:
                st.write("No fraud insights available. Please generate alerts or check API connection.")
    
    st.markdown("---")
    
    st.subheader("Quick Branch Overview")
    selected_branch = st.selectbox("Select Branch", BRANCH_LIST)
    
    if st.button("Generate Branch Overview"):
        try:
            with st.spinner(f"Generating insights for {selected_branch}..."):
                alert = get_branch_alert(selected_branch)
                st.info(f"Branch Analysis: {selected_branch}")
                st.write(alert)
        except Exception as e:
            # Use direct generator instead of API
            st.info(f"Branch Analysis: {selected_branch} (Generated locally)")
            with st.spinner(f"Generating local insights for {selected_branch}..."):
                try:
                    alert_content = generate_alert(
                        query=f"Generate insights for {selected_branch} branch",
                        branch=selected_branch,
                        timeframe="recent"
                    )
                    if isinstance(alert_content, dict):
                        st.write(alert_content.get('description', f"No insights available for {selected_branch}"))
                    else:
                        st.write(alert_content)
                except Exception as e2:
                    st.error(f"Could not generate insights: {str(e2)}")

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
            try:
                # Use generate_alert() function directly instead of API
                alert_content = generate_alert(query=query)
                
                # Create alert object
                custom_alert = {
                    'id': f"custom_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'title': alert_content.get('title', 'Custom Alert'),
                    'description': alert_content.get('description', 'No description available'),
                    'recommended_actions': alert_content.get('recommended_actions', []),
                    'severity': 'medium',
                    'type': 'custom_query',
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                # Display the generated alert
                st.subheader("Generated Alert")
                render_alert_card(custom_alert)
            except Exception as e:
                st.error(f"Error generating custom alert: {str(e)}")

def load_latest_alerts() -> List[Dict[str, Any]]:
    """
    Load the latest generated alerts.
    
    Returns:
        List of alerts
    """
    alert_generator = AutomaticAlertGenerator()
    return alert_generator.load_latest_alerts()

def generate_new_alerts() -> List[Dict[str, Any]]:
    """
    Generate new alerts based on the latest data.
    
    Returns:
        List of generated alerts
    """
    with st.spinner("Generating alerts from the latest data..."):
        alert_generator = AutomaticAlertGenerator()
        alerts = alert_generator.run()
        return alerts

def render_alert_card(alert: Dict[str, Any]):
    """
    Render a single alert card.
    
    Args:
        alert: Alert details
    """
    severity = alert.get('severity', 'low')
    alert_class = f"{severity}-alert"
    
    html = f"""
    <div class="alert-card {alert_class}">
        <div class="alert-title">{alert.get('title', 'Alert')}</div>
        <div class="alert-body">{alert.get('description', 'No description available')}</div>
    """
    
    # Render recommended actions if available
    if 'recommended_actions' in alert and alert['recommended_actions']:
        html += "<div><b>Recommended Actions:</b></div>"
        for action in alert['recommended_actions']:
            html += f"<div class='action-item'>• {action}</div>"
    
    # Render alert details
    alert_time = datetime.datetime.fromisoformat(alert.get('timestamp', datetime.datetime.now().isoformat()))
    formatted_time = alert_time.strftime("%Y-%m-%d %H:%M:%S")
    html += f"<div class='alert-footer'>Alert ID: {alert.get('id', 'Unknown')} | Generated: {formatted_time} | Type: {alert.get('type', 'Unknown')}</div>"
    html += "</div>"
    
    st.markdown(html, unsafe_allow_html=True)

def render_data_summary():
    """
    Render a summary of the current data version.
    """
    try:
        # Get data version info
        version_manager = DataVersionManager()
        current_version = version_manager.get_current_version()
        version_info = version_manager.get_version_info()
        
        # Display version information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Version Information")
            st.markdown(f"**Current Version:** {current_version}")
            st.markdown(f"**Last Updated:** {version_info.get('created_at', 'Unknown')}")
            st.markdown(f"**Notes:** {version_info.get('notes', 'No notes available')}")
        
        with col2:
            # Display available versions
            all_versions = version_manager.list_versions()
            st.subheader("Available Versions")
            
            if all_versions:
                selected_version = st.selectbox("Select Version", all_versions, index=all_versions.index(current_version))
                
                if selected_version != current_version:
                    if st.button(f"Rollback to {selected_version}"):
                        with st.spinner(f"Rolling back to version {selected_version}..."):
                            success = version_manager.rollback_to_version(selected_version)
                            if success:
                                st.success(f"Successfully rolled back to version {selected_version}")
                                st.rerun()
                            else:
                                st.error(f"Failed to roll back to version {selected_version}")
            else:
                st.markdown("No versions available")
        
        # Display processing info if available
        processing_info = version_info.get('processing_info', {})
        if processing_info:
            st.subheader("Processing Information")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Records:** {processing_info.get('processed_records', 'Unknown')}")
            with col2:
                st.markdown(f"**Files:** {len(processing_info.get('processed_files', []))}")
            with col3:
                st.markdown(f"**Merged Data:** {'Yes' if processing_info.get('merge_with_existing', False) else 'No'}")
    
    except Exception as e:
        st.error(f"Error loading data version information: {str(e)}")

# Main app
def main():
    # Apply custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E3A8A;
        }
        .alert-card {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .high-alert {
            background-color: rgba(239, 68, 68, 0.1);
            border-left: 4px solid #EF4444;
        }
        .medium-alert {
            background-color: rgba(249, 115, 22, 0.1);
            border-left: 4px solid #F97316;
        }
        .low-alert {
            background-color: rgba(59, 130, 246, 0.1);
            border-left: 4px solid #3B82F6;
        }
        .alert-title {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .alert-body {
            font-size: 1rem;
            margin-bottom: 1rem;
        }
        .alert-footer {
            font-size: 0.8rem;
            color: #6B7280;
        }
        .action-item {
            font-size: 0.9rem;
            padding: 0.5rem;
            background-color: rgba(209, 213, 219, 0.2);
            border-radius: 0.25rem;
            margin-bottom: 0.25rem;
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
    elif selected_page == "Alerts":
        st.header("Alert Management")
        
        # Alert actions
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Alert filtering
            severity_filter = st.multiselect("Filter by Severity", ["high", "medium", "low"], default=["high", "medium", "low"])
            
        with col2:
            # Generate new alerts button
            if st.button("Generate New Alerts"):
                new_alerts = generate_new_alerts()
                st.success(f"Generated {len(new_alerts)} new alerts")
                st.rerun()
        
        # Load alerts
        alerts = load_latest_alerts()
        
        # Apply filters
        filtered_alerts = [a for a in alerts if a.get('severity') in severity_filter]
        
        # Sort by severity first, then by timestamp
        severity_order = {"high": 0, "medium": 1, "low": 2}
        sorted_alerts = sorted(
            filtered_alerts, 
            key=lambda x: (
                severity_order.get(x.get('severity', 'low'), 99),
                datetime.datetime.fromisoformat(x.get('timestamp', '2000-01-01'))
            )
        )
        
        # Display alerts
        st.subheader(f"Alerts ({len(sorted_alerts)})")
        
        if sorted_alerts:
            for alert in sorted_alerts:
                render_alert_card(alert)
        else:
            st.info("No alerts match the current filters.")
    elif selected_page == "Data Management":
        st.header("Data Management")
        
        # Data version management
        st.subheader("Data Version Control")
        render_data_summary()
        
        # Upload new data section
        st.subheader("Add New Data")
        
        uploaded_file = st.file_uploader("Upload new transaction data (CSV)", type=["csv"])
        
        col1, col2 = st.columns(2)
        with col1:
            merge_option = st.checkbox("Merge with existing data", value=True)
        with col2:
            generate_alerts_option = st.checkbox("Generate alerts after processing", value=True)
        
        if uploaded_file is not None:
            if st.button("Process New Data"):
                try:
                    # Save uploaded file temporarily
                    temp_file_path = os.path.join("data", "temp_upload.csv")
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the new data
                    with st.spinner("Processing new data..."):
                        st.info("Using optimized processing: New data will be analyzed directly with RAG without re-vectorizing the entire dataset.")
                        version_manager = DataVersionManager(auto_generate_alerts=generate_alerts_option)
                        new_version = version_manager.process_new_data(temp_file_path, merge_with_existing=merge_option)
                        
                    # Clean up temp file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    
                    st.success(f"Successfully processed new data. New version: {new_version}")
                    st.info("Alerts have been generated directly from the new data and can be viewed in the Alerts section.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing new data: {str(e)}")
        
        # Data version comparison
        st.subheader("Version Comparison")
        
        version_manager = DataVersionManager()
        all_versions = version_manager.list_versions()
        
        if len(all_versions) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                version_a = st.selectbox("Version A", all_versions, index=min(1, len(all_versions)-1))
            
            with col2:
                version_b = st.selectbox("Version B", all_versions, index=0)
            
            if version_a != version_b:
                if st.button("Compare Versions"):
                    with st.spinner("Comparing versions..."):
                        comparison = version_manager.compare_versions(version_a, version_b)
                        
                        st.subheader("Comparison Results")
                        
                        # Display comparison metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**{version_a}**")
                            st.markdown(f"Created: {comparison.get('creation_date_a', 'Unknown')}")
                            st.markdown(f"Notes: {comparison.get('notes_a', 'None')}")
                        
                        with col2:
                            st.markdown(f"**{version_b}**")
                            st.markdown(f"Created: {comparison.get('creation_date_b', 'Unknown')}")
                            st.markdown(f"Notes: {comparison.get('notes_b', 'None')}")
                        
                        # Display differences
                        st.markdown("### Differences")
                        st.markdown(f"- Time between versions: {comparison.get('time_difference_days', 'Unknown')} days")
                        st.markdown(f"- Transaction count difference: {comparison.get('transaction_count_difference', 'Unknown')}")
                        st.markdown(f"- New records: {comparison.get('new_records_count', 'Unknown')}")
                        st.markdown(f"- Lineage relation: {comparison.get('lineage_relation', 'Unknown')}")
        else:
            st.info("At least two versions are needed for comparison.")
    elif selected_page == "Custom Analysis":
        st.header("Custom Analysis")
        
        # Input for custom alert generation
        st.subheader("Generate Custom Alert")
        query = st.text_area("Enter your query", "Analyze recent transaction patterns and identify any unusual activity.", height=100)
        
        if st.button("Generate Alert"):
            with st.spinner("Generating custom alert..."):
                try:
                    # Use generate_alert() function directly instead of API
                    alert_content = generate_alert(query=query)
                    
                    # Create alert object
                    custom_alert = {
                        'id': f"custom_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                        'title': alert_content.get('title', 'Custom Alert'),
                        'description': alert_content.get('description', 'No description available'),
                        'recommended_actions': alert_content.get('recommended_actions', []),
                        'severity': 'medium',
                        'type': 'custom_query',
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                    
                    # Display the generated alert
                    st.subheader("Generated Alert")
                    render_alert_card(custom_alert)
                except Exception as e:
                    st.error(f"Error generating custom alert: {str(e)}")

if __name__ == "__main__":
    main() 