"""
One-command script to run the entire AI-powered Alert Management System.

This script:
1. Processes raw data
2. Generates embeddings
3. Starts the API server
4. Launches the Streamlit UI
"""

import os
import subprocess
import sys
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor

def run_command(command, name, new_window=False):
    """Run a command and print its output."""
    print(f"Starting {name}...")
    
    if new_window:
        if os.name == 'nt':  # Windows
            subprocess.Popen(f'start cmd /k "{command}"', shell=True)
        else:  # Unix-like systems
            subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', command + '; exec bash'])
    else:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Print first few lines of output
        for _ in range(5):
            line = process.stdout.readline()
            if not line:
                break
            print(f"{name}: {line.strip()}")
        
        return process

def process_data():
    """Process raw transaction data."""
    print("Step 1: Processing raw data...")
    result = subprocess.run(
        "python data/processed/data_processor.py",
        shell=True,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"Error processing data: {result.stderr}")
        sys.exit(1)
    
    print("Data processing completed successfully.")

def generate_embeddings():
    """Generate embeddings for processed data."""
    print("Step 2: Generating embeddings...")
    result = subprocess.run(
        "python embeddings/embedder.py",
        shell=True,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"Error generating embeddings: {result.stderr}")
        sys.exit(1)
    
    print("Embeddings generation completed successfully.")

def start_api_server():
    """Start the FastAPI server."""
    print("Step 3: Starting API server...")
    api_cmd = "uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"
    return run_command(api_cmd, "API Server", new_window=True)

def start_streamlit_ui():
    """Start the Streamlit UI."""
    print("Step 4: Starting Streamlit UI...")
    ui_cmd = "streamlit run app.py"
    return run_command(ui_cmd, "Streamlit UI", new_window=True)

def open_browser():
    """Open browser tabs for UI and API docs."""
    print("Opening browser tabs...")
    time.sleep(5)  # Give servers time to start
    
    try:
        webbrowser.open("http://localhost:8501")  # Streamlit UI
        time.sleep(1)
        webbrowser.open("http://localhost:8000/docs")  # API docs
    except Exception as e:
        print(f"Error opening browser: {e}")

def main():
    """Run the entire system."""
    print("=" * 50)
    print("Starting AI-powered Alert Management System")
    print("=" * 50)
    
    # Sequential steps for data processing
    process_data()
    generate_embeddings()
    
    # Start servers in parallel
    with ThreadPoolExecutor() as executor:
        api_future = executor.submit(start_api_server)
        time.sleep(2)  # Give API server time to start
        ui_future = executor.submit(start_streamlit_ui)
        
        # Wait for servers to start
        time.sleep(3)
        
        # Open browser tabs
        open_browser()
    
    print("\nSystem is now running!")
    print("- Streamlit UI: http://localhost:8501")
    print("- API Server: http://localhost:8000")
    print("- API Documentation: http://localhost:8000/docs")
    print("\nPress Ctrl+C in the terminal windows to stop the servers.")

if __name__ == "__main__":
    main() 