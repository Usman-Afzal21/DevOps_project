# Data Versioning System with DVC

This document describes the data versioning system implemented for the AI-powered Alert Management System. The system automatically updates the RAG vector store whenever the underlying data changes.

## Overview

The data versioning system uses:
- **DVC (Data Version Control)** for tracking data files
- **Custom hooks** to automatically update the vector store when data changes
- **GitHub Actions** for CI/CD pipeline integration

## Components

### 1. DataVersionManager

The `DataVersionManager` class in `data/data_versioning.py` handles:
- Data version control
- Dataset updates with new data
- Tracking data lineage 
- Integration with the RAG pipeline
- Vector store updates

### 2. DVC Setup

DVC is used to track:
- Raw data (`data/raw/`)
- Processed data (`data/processed/`)
- Vector database (`data/vector_db/`)

### 3. Automatic Vector Store Updates

The system automatically updates the vector store when data changes through:
- DVC hooks that trigger after checkout, commit, or pull operations
- GitHub Actions that run when data files are modified

## How to Use

### Adding New Data

1. Place new data files in the `data/raw/` directory
2. Use the DataVersionManager to process and integrate the new data:

```python
from data.data_versioning import DataVersionManager

# Initialize version manager
version_manager = DataVersionManager()

# Process new data file - this automatically updates embeddings and vector store
new_version = version_manager.process_new_data("path/to/new_data.csv")
```

3. Update DVC tracking (done automatically if `core.autostage` is enabled):

```bash
dvc add data/raw/ data/processed/ data/vector_db/
```

### Managing Versions

List available versions:
```python
versions = version_manager.list_versions()
print(f"Available versions: {versions}")
```

Compare versions:
```python
comparison = version_manager.compare_versions("v1.0.0", "v2.0.0")
```

Rollback to a previous version:
```python
version_manager.rollback_to_version("v1.0.0")
```

## Automated Workflows

### Local Hooks

DVC hooks in `.dvc/hooks/` directory automatically update the vector store:
- `post-checkout`: After checking out data
- `post-commit`: After committing data changes
- `post-pull`: After pulling data from remote storage

### CI/CD Integration

GitHub Actions workflow in `.github/workflows/update_vector_store.yml` automatically:
1. Detects changes to data files
2. Updates the vector store
3. Commits the changes back to the repository

## Testing

Run the test script to verify the system is working correctly:

```bash
python test_dvc_update.py
```

This will:
1. Create a new data version
2. Update the vector store
3. Verify that the updates were successful
4. Update DVC tracking 