# AI-powered Banking Alert Management System

An MLOps project for a bank's AI-powered Alert Management System using RAG with Llama 3 via the Groq API.

## Overview

The system processes bank transaction data and other banking datasets, generates embeddings, and provides various alerts through API endpoints with a visual Streamlit interface.

Key components include:
- Data processing pipeline for multiple datasets
- Embeddings generation using Sentence Transformers
- Vector store using ChromaDB
- RAG pipeline with Llama 3 via Groq
- API for alert generation
- Streamlit UI for visualizing alerts
- **Data versioning with DVC**
- **Automated alert generation system**
- **CI/CD pipeline with GitHub Actions**

## Key Features

### Data Versioning System
- Data version tracking with semantic versioning
- Metadata storage and lineage tracking
- Ability to process and integrate new data
- Version rollback and comparison functionality
- Automated pipeline updates with DVC

### Automated Alert Generation
- **Fully automated alert detection & generation**
- Analysis of all banking datasets (not just transactions)
- Severity classification and prioritization
- RAG-powered alert descriptions and recommendations
- Automatic generation when data changes

## Running the System

### Prerequisites
- Python 3.11+
- DVC installed (`pip install dvc`)
- Groq API key (set in .env file)

### Start the Application
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start the API server:
   ```
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. Start the Streamlit UI:
   ```
   streamlit run app.py
   ```

## Adding New Data

### Option 1: Via Streamlit UI
1. Navigate to the "Data Management" page
2. Upload your CSV file
3. Choose merge options
4. Click "Process New Data"

### Option 2: Via Command Line
```
python add_new_data.py path/to/new_data.csv
```

The system will:
1. Process and integrate the new data
2. Generate embeddings
3. Update the vector store
4. Automatically generate alerts for any anomalies

## Using DVC for Data Version Control

### Basic Commands
```
dvc add data/raw/new_file.csv  # Track a new file
dvc push                       # Push to remote storage
dvc pull                       # Pull from remote storage
```

DVC hooks automatically:
1. Update the vector store
2. Generate new alerts when data changes

## Documentation

- [Data Versioning System](DATA_VERSIONING.md)
- [Alert Generation System](ALERT_SYSTEM.md)

## Project Structure

- `api/`: FastAPI endpoints
- `data/`: Data files and versioning
  - `raw/`: Raw banking datasets
  - `processed/`: Processed data files
  - `versions/`: Versioned data snapshots
  - `vector_db/`: ChromaDB vector database
- `embeddings/`: Embedding generation and management
- `rag_pipeline/`: RAG implementation with Groq
- `ci_cd/`: CI/CD configuration files
- `alerts/`: Generated alert JSON files
- `app.py`: Streamlit UI
- `add_new_data.py`: Script for adding new data
- `update_vector_store.py`: Script for updating the vector store

# Banking RAG System: Optimized Data Processing

## About this Project
This project implements a Retrieval-Augmented Generation (RAG) system for a banking application.
It uses LLama 3 via Groq API to analyze banking transaction data, detect anomalies, and generate intelligent alerts.

## Optimized RAG Approach
The system uses an optimized approach for handling data:

1. **One-time Vector Store Initialization:** 
   - The base data in `data/processed` is vectorized only once during setup
   - This provides the foundation for RAG functionality without repeated processing
   
2. **Direct Analysis of New Data:**
   - New data uploaded through the UI is analyzed directly without re-vectorizing
   - The system uses the existing vector store for context through RAG
   - This is significantly more computationally efficient for large datasets

3. **Versioning Only New Data:**
   - Only the new data is versioned with DVC, not the vector store
   - This reduces storage requirements and improves performance

## System Setup

### First-time Setup
Run the setup script to initialize the system with base data:
```
python setup_rag_system.py
```

This will:
1. Process the base data in `data/raw`
2. Initialize the vector store once with the base data
3. Set up the system for direct data analysis

### Running the System
Start the system with:
```
update_vector_store.bat
```

Or manually start each component:
1. Start the API server: `python api/main.py`
2. Start the UI: `streamlit run app.py`

### Processing New Data
Upload new data through the UI:
1. Go to "Data Management" tab
2. Upload a CSV file
3. Click "Process New Data"

The system will:
- Analyze the new data directly without re-vectorizing
- Generate alerts based on detected anomalies
- Display insights using the RAG approach

## Project Structure
- `app.py`: Streamlit UI for the banking dashboard
- `api/`: API server code
- `data/`: Transaction data (raw and processed)
- `embeddings/`: Embedding generation code
- `rag_pipeline/`: RAG implementation
- `initialize_vector_store.py`: Script to initialize the vector store once
- `rag_pipeline/direct_data_analyzer.py`: Efficient direct data analysis

## Technical Details
- The system uses a custom embedding function to avoid ONNX dependencies
- Alerts are generated using RAG with LLama 3 via Groq API
- DVC is used for versioning input data only, not for the vector store