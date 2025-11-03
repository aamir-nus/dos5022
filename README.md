# dos5022
DOS5022 Assignment - Tender Data Analysis

## Project Overview

This project analyzes Singapore government tender data to provide insights on:
1. **Award Amount Prediction**: Predict tender award amounts from text and features
2. **Award Status Classification**: Classify likelihood of "Awarded" vs "No award"
3. **Clustering Analysis**: Group agencies/suppliers by spend patterns for category strategy and risk identification

## Data Source

Dataset: [Government Procurement Data](https://data.gov.sg/datasets/d_acde1106003906a75c3fa052592f2fcb/view)

Note: If the network is restricted, the system will generate sample data automatically.

## Project Structure

```
dos5022/
├── src/
│   ├── __init__.py
│   ├── download_data.py       # Data acquisition
│   ├── preprocessing.py       # Data cleaning and feature engineering
│   ├── visualization.py       # Exploratory data analysis
│   └── models/
│       ├── __init__.py
│       ├── award_predictor.py      # Regression models for award amounts
│       ├── award_classifier.py     # Classification models for award status
│       └── clustering.py           # Clustering for spend patterns
├── data/                      # Data files (auto-generated, gitignored)
├── main.py                    # Main execution script
├── pyproject.toml            # Project dependencies (uv)
└── README.md                 # This file
```

## Setup

This project uses `uv` for Python package management with Python 3.10.

### Install Dependencies

```bash
# Install uv if not already installed
pip install uv

# Install project dependencies
uv sync
```

## Usage

### Run Complete Pipeline

Execute the entire analysis pipeline:

```bash
uv run python main.py
```

This will:
1. Download/generate tender data
2. Preprocess and engineer features
3. Generate visualizations
4. Train prediction models (Random Forest, Gradient Boosting, Ridge)
5. Train classification models (Random Forest, Gradient Boosting, Logistic Regression)
6. Perform clustering analysis
7. Identify risk flags

### Run Individual Modules

```bash
# Download data
uv run python src/download_data.py

# Preprocess data
uv run python src/preprocessing.py

# Generate visualizations
uv run python src/visualization.py

# Train prediction models
uv run python src/models/award_predictor.py

# Train classification models
uv run python src/models/award_classifier.py

# Perform clustering analysis
uv run python src/models/clustering.py
```

## Outputs

All outputs are saved in the `data/` directory:

- **tender_data.csv**: Raw tender data
- **tender_data_processed.csv**: Processed data with engineered features
- **visualizations/**: EDA plots
  - award_distribution.png
  - award_amounts_distribution.png
  - agency_analysis.png
  - category_analysis.png
  - temporal_trends.png
  - correlation_heatmap.png
- **Models**: Trained ML models (.pkl files)
- **agency_clusters.csv**: Agency clustering results
- **supplier_clusters.csv**: Supplier clustering results
- **risk_flags.csv**: Identified risk flags

## Features

### Preprocessing
- Data cleaning and validation
- Date feature engineering
- Text feature extraction
- Categorical encoding
- Feature scaling

### Visualizations
- Award status distribution
- Award amount distributions
- Agency-level analysis
- Category-level analysis
- Temporal trends
- Feature correlations

### Models

#### 1. Award Amount Prediction
- **Models**: Random Forest, Gradient Boosting, Ridge Regression
- **Target**: Predicting tender award amounts
- **Features**: Agency, category, date features, text features
- **Evaluation**: RMSE, MAE, R²

#### 2. Award Status Classification
- **Models**: Random Forest, Gradient Boosting, Logistic Regression
- **Target**: Classifying "Awarded" vs "No award"
- **Evaluation**: Accuracy, Precision, Recall, F1, ROC AUC

#### 3. Clustering Analysis
- **Method**: K-Means clustering
- **Entities**: Agencies and Suppliers
- **Features**: Spend patterns, diversity metrics
- **Output**: Risk flags and insights

## Risk Flags

The system identifies potential risks:
- Agencies with low supplier diversity (dependency risk)
- Agencies with low category diversity (missed opportunities)
- Suppliers with single agency dependency (business risk)
- Suppliers with single category focus (limited capability)

## Requirements

- Python 3.10+
- uv package manager
- Dependencies managed in pyproject.toml

## License

See LICENSE file for details.

