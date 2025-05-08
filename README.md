# AI-powered Alert Management System on MIS Platform

An MLOps project implementing a Retrieval-Augmented Generation (RAG) system for generating intelligent alerts based on banking transaction data.

## Project Overview

This project builds an AI solution that configures alerts on banking MIS (Management Information System) reports based on historical data. It uses a RAG architecture to combine the power of LLMs with retrieval from banking transaction data.

## Features

- Data processing pipeline for bank transaction data
- Vector embeddings for RAG functionality
- Integration with Llama 3.2/3.3 via Groq API
- Intelligent alert generation based on transaction patterns
- Full MLOps lifecycle implementation
- Streamlit UI for interactive alert exploration
- **Data versioning and automated pipeline updates**

## Project Structure

```
project/
├── data/
│   ├── raw/               # Original transaction data
│   └── processed/         # Processed and vectorized data
├── embeddings/            # Vector embeddings for RAG 
├── rag_pipeline/          # RAG implementation components
├── api/                   # FastAPI server for alert generation
├── notebooks/             # Development notebooks
├── tests/                 # Test suite
├── ci_cd/                 # CI/CD pipeline configuration
├── monitoring/            # Monitoring and logging setup
├── app.py                 # Streamlit UI application
├── run_system.py          # One-command system launcher
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.9+
- Groq API key for LLM access

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   pip install dvc
   ```
3. Set up environment variables:
   ```
   export GROQ_API_KEY=your_api_key_here
   ```

### Running the System

#### Option 1: One-Command Launch (Recommended)

Run the entire system with a single command:

```
python run_system.py
```

This will:
1. Process the raw data
2. Generate embeddings
3. Start the API server
4. Launch the Streamlit UI
5. Open browser tabs for the UI and API documentation

#### Option 2: Step-by-Step Launch

To run the complete system manually, follow these steps in order:

1. **Process Raw Data**:
   ```
   python data/processed/data_processor.py
   ```

2. **Generate Embeddings**:
   ```
   python embeddings/embedder.py
   ```

3. **Start the API Server**:
   ```
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Launch the Streamlit UI**:
   ```
   streamlit run app.py
   ```

The Streamlit UI will be available at http://localhost:8501, and the API server will be accessible at http://localhost:8000.

## MLOps Lifecycle

This project implements the complete MLOps lifecycle:

1. **Data Engineering**
   - Data ingestion from raw transaction data
   - Data preprocessing and feature engineering
   - Data versioning

2. **Model Development**
   - Vector embeddings generation
   - RAG system implementation with Llama 3 via Groq
   - Evaluation and testing

3. **Deployment**
   - FastAPI service for alert generation
   - Containerization with Docker
   - Streamlit UI for business users

4. **Monitoring**
   - Performance metrics tracking
   - Model drift detection
   - Alert quality assessment

5. **CI/CD**
   - Automated testing
   - Continuous deployment

## Usage

### API Usage

To generate alerts using the API directly:

```python
from rag_pipeline.generator import generate_alert

# Generate alert for specific criteria
alert = generate_alert(
    query="What unusual patterns are in recent transactions?",
    timeframe="last_week"
)
print(alert)
```

### UI Usage

The Streamlit UI provides an intuitive interface for:
- General alerts dashboard
- Branch-specific analysis
- Fraud detection
- Comparative analysis between branches
- Trend analysis over time periods
- Custom alert generation

## License

This project is licensed under the MIT License.

## Data Versioning System

We've implemented a comprehensive data versioning system using DVC (Data Version Control). This system:
- Tracks all data files with proper versioning
- Automatically updates vector stores when data changes
- Maintains data lineage and provenance
- Allows rollback to previous versions
- Integrates with CI/CD via GitHub Actions

For detailed information on the data versioning system, see [DATA_VERSIONING.md](DATA_VERSIONING.md).

## Adding New Data

To add new transaction data and automatically update the vector store:

```bash
python add_new_data.py path/to/new_transaction_data.csv
```

This will:
1. Process the new data
2. Create a new version
3. Update embeddings
4. Update the vector store
5. Update DVC tracking

## Project Structure

- `api/`: API endpoints for alert generation
- `data/`: Data processing and versioning
  - `raw/`: Raw transaction data
  - `processed/`: Cleaned and processed data
  - `vector_db/`: ChromaDB vector database
- `embeddings/`: Embeddings generation and vector store
- `rag_pipeline/`: RAG implementation with Groq API
- `monitoring/`: Monitoring and logging
- `ci_cd/`: CI/CD configuration
- `.dvc/`: DVC configuration and hooks
- `.github/workflows/`: GitHub Actions workflows

## MLOps Features

- **Data Version Control**: DVC for tracking data changes
- **Automated Pipeline**: Automatic updates when data changes
- **Data Lineage**: Tracking data sources and transformations
- **CI/CD Integration**: GitHub Actions for automation
- **Monitoring**: Logging and performance tracking
- **Fallback Mechanisms**: Graceful degradation when components fail