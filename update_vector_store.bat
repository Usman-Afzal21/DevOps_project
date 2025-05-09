@echo off
echo ===============================
echo Banking RAG System Startup
echo ===============================

:: Activate virtual environment if needed
:: Uncomment the following line if you have a virtual environment
:: call venv\Scripts\activate.bat

echo [1/3] Starting API server on port 8001...
start /B python api/main.py

:: Wait for API to start up
echo Waiting for API to initialize...
timeout /t 3 /nobreak > nul

:: Test API health
echo Checking API health...
curl -s http://localhost:8001/health

echo [2/3] Processing test data with optimized direct analysis...
python rag_pipeline/direct_data_analyzer.py enhanced_test_data.csv

echo [3/3] Starting Streamlit UI on port 8502...
start /B streamlit run app.py

echo ===============================
echo System startup complete!
echo ===============================
echo.
echo Important Notes:
echo - New approach: Data is analyzed directly without re-vectorizing
echo - Vector store is only initialized once with base data
echo - Upload new data through the UI for efficient processing
echo - To initialize the system for the first time: python setup_rag_system.py
echo.
echo Visit http://localhost:8502 to access the dashboard 