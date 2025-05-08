# Automatic Alert Generation System

## Overview

The Automatic Alert Generation System is a core component of the Banking Alert Management System. It analyzes multiple banking datasets and automatically generates alerts when potential issues or anomalies are detected. Alerts are intelligently created using RAG (Retrieval-Augmented Generation) with Llama 3 via the Groq API.

## Key Features

- **Fully Automated**: Generates alerts without manual prompting whenever data changes
- **Multi-Dataset Analysis**: Processes all banking datasets, not just transaction data
- **Anomaly Detection**: Identifies unusual patterns across multiple data dimensions
- **Severity Classification**: Categorizes alerts as high, medium, or low priority
- **RAG-Powered Insights**: Uses LLM context to provide detailed explanations and recommendations
- **Integration with Data Versioning**: Automatically triggered after data updates

## Alert Types

The system can detect and generate alerts for various anomalies, including:

- Large transactions exceeding thresholds
- Potential fraud indicators
- Unusual login activity
- High-volume transaction patterns
- Branch-level anomalies
- Extreme values in any numerical field
- Any flagged records across datasets

## How It Works

1. **Data Processing**: 
   - The system processes all raw datasets in `data/raw/`
   - Data is cleaned, preprocessed, and features are engineered
   - Processed data is stored in `data/processed/`

2. **Anomaly Detection**:
   - Statistical analysis to identify outliers and unusual patterns
   - Domain-specific rules to detect banking-relevant anomalies
   - Comparative analysis against historical data

3. **Alert Generation**:
   - RAG pipeline generates detailed alerts with context
   - Each alert includes a title, description, and recommended actions
   - Alerts are stored as JSON files in the `alerts/` directory

4. **Integration with UI**:
   - Alerts are displayed in the Streamlit interface
   - Filterable by severity and alert type
   - Dashboard summarizes alert status

## Trigger Points

Alerts are automatically generated whenever:

1. New data is added via `add_new_data.py`
2. The vector store is updated via `update_vector_store.py`
3. DVC detects changes and runs post-checkout, post-commit, or post-pull hooks
4. The CI/CD pipeline runs the GitHub Actions workflow
5. A data version rollback occurs

## Using the Alert System

### Viewing Alerts

1. Start the Streamlit UI: `streamlit run app.py`
2. Navigate to the "Dashboard" to see high-priority alerts
3. Go to the "Alerts" page to see all alerts with filtering options

### Generating New Alerts

Alerts are automatically generated when:
- New data is uploaded through the UI's "Data Management" page
- New data is processed with `python add_new_data.py new_data.csv`
- Data is pulled or updated with DVC: `dvc pull` or `dvc checkout`

You can also manually trigger alert generation:
- Click "Generate New Alerts" on the Alerts page
- Run `python -c "from rag_pipeline.alert_generator import AutomaticAlertGenerator; AutomaticAlertGenerator().run()"`

### Custom Alerts

You can generate custom alerts by:
1. Going to the "Custom Analysis" page
2. Entering your specific query
3. Clicking "Generate Alert"

## Alert Structure

Each alert contains:

```json
{
  "id": "alert_1_20250508120000",
  "title": "Large Transaction Alert: 15 Unusual Transactions Detected",
  "description": "15 transactions exceeding the threshold of $15,000 were detected...",
  "recommended_actions": [
    "Review these transactions for legitimacy",
    "Contact customers for verification",
    "Update transaction monitoring thresholds"
  ],
  "severity": "medium",
  "type": "large_transactions",
  "timestamp": "2025-05-08T12:00:00",
  "status": "new",
  "anomaly_data": {
    // Detailed anomaly information
  }
}
```

## Customization

The alert generation system can be customized by:

1. **Adjusting Thresholds**: Modify the `alert_threshold` parameter in `AutomaticAlertGenerator`
2. **Adding New Detection Rules**: Extend the `detect_anomalies` method in `alert_generator.py`
3. **Customizing RAG Queries**: Modify the `_create_anomaly_query` method to change how alerts are generated
4. **Setting Alert Severity**: Adjust the severity assignment logic in the detection code

## Files and Components

- `rag_pipeline/alert_generator.py`: Core alert generation logic
- `data/data_versioning.py`: Integration with data versioning system
- `update_vector_store.py`: Script that updates vectors and triggers alerts
- `app.py`: Streamlit UI for displaying and interacting with alerts
- `alerts/`: Directory storing generated alert JSON files 