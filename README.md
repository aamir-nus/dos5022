# Singapore Government Tender Analysis

A comprehensive data science pipeline for analyzing Singapore government procurement data through advanced machine learning techniques and business intelligence.

## Executive Summary

Singapore's government procurement ecosystem represents a S$40+ billion annual marketplace serving 111 agencies across 6,000+ suppliers. This project transforms raw procurement data into strategic intelligence, enabling data-driven decision-making that enhances procurement efficiency, manages supplier relationships, and identifies commercial risks.

**Key Achievements:**

- **87% accuracy** in commercial risk prediction using machine learning
- **3 distinct supplier segments** identified through clustering analysis

## Business Context & Impact

### Industry Background

Singapore operates one of the world's most transparent and efficient public procurement systems through the GeBIZ (Government Electronic Business) platform. This analysis supports:

- **Strategic Supplier Portfolio Optimization** - Balance cost efficiency, risk management, and SME development
- **Predictive Commercial Risk Assessment** - Proactive identification of unusual award amounts requiring review
- **Market Intelligence & Competition Analysis** - Understanding supplier concentration and diversity dynamics

### Key Business Questions Answered

1. Which agencies dominate government procurement and what are their spending patterns?
2. How concentrated is the supplier base and what are the associated risks?
3. Which procurement methods are most effective and how do they correlate with outcomes?
4. What temporal patterns exist in government spending and award timing?
5. How can we identify unusual awards that warrant commercial review?

## Dataset Overview

**Source:** [Government Procurement via GeBIZ](https://data.gov.sg/datasets/d_acde1106003906a75c3fa052592f2fcb/view) (2020-2025)

**Scale:**

- **18,021** tender records
- **111** unique government agencies
- **6,083** unique suppliers
- **S$0 to S$1.49B** award amount range
- **96.1%** award success rate

**Key Fields:**

- `tender_no` - Unique tender identifier
- `tender_description` - Procurement requirements (13-500 characters)
- `agency` - Procuring government agency
- `supplier_name` - Awarded supplier
- `awarded_amt` - Contract value in SGD
- `award_date` - Award decision date
- `tender_detail_status` - Procurement method used

## Installation & Setup

### Prerequisites

- **Python 3.10+** required
- **uv** package manager (recommended) or pip

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/aamir-nus/dos5022.git
cd dos5022

# Activate virtual environment (if using uv)
source .venv/bin/activate

# Install dependencies
uv sync --all-groups

# Verify installation
uv run python -c "import pandas, sklearn, yaml; print('All dependencies installed')"
```

## Usage Guide

### 1. Interactive Analysis (Recommended)

**Start with the preprocessing notebook:**

```bash
jupyter notebook notebooks/01_preprocessing.ipynb
```

**Continue with modeling:**

```bash
jupyter notebook notebooks/02_modeling.ipynb
```

**Business insights and analysis:**

```bash
jupyter notebook notebooks/supplier clustering and risk prediction.ipynb
```

### 2. Configuration-Driven Pipeline

**Run the complete preprocessing pipeline:**

```bash
uv run python -c "
from src.preprocessing import preprocess_pipeline
preprocessor, df_processed = preprocess_pipeline()
print(f'✓ Processed {df_processed.shape[0]} tenders with {df_processed.shape[1]} features')
"
```

## Trained Models & Usage

### Model File Locations

All trained models are saved in the **`models/`** directory after running the notebooks:

#### Risk Prediction Model (Classification)

- **`commercial_risk_predictor.pkl`** - Random Forest Classifier for commercial review prediction

#### Supplier Clustering Model

- **`supplier_clustering_model.pkl`** - K-Means clustering for supplier portfolio optimization

#### Preprocessing Pipeline

- **`preprocessor.pkl`** - Complete preprocessing pipeline for new data transformation

### Using Trained Models

```python
import pickle
import pandas as pd
from src.preprocessing import preprocess_pipeline

# Load preprocessing pipeline and models
with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('models/commercial_risk_predictor.pkl', 'rb') as f:
    risk_model = pickle.load(f)

# Prepare new tender data
new_tender_data = pd.DataFrame({
    'tender_description': ['IT infrastructure services'],
    'agency': ['Ministry of Health'],
    'award_date': ['2024-12-01'],
    'supplier_name': ['Tech Solutions Pte Ltd'],
    'awarded_amt': [5000000],  # High-value award for risk assessment
    'tender_detail_status': ['Awarded to Suppliers']
})

# Preprocess and predict risk level
X_processed = preprocessor.transform(new_tender_data)
risk_prediction = risk_model.predict(X_processed)
risk_probability = risk_model.predict_proba(X_processed)

print(f'Risk Level: {risk_prediction[0]}')
print(f'High Risk Probability: {risk_probability[0][1]:.1%}')
```

### Key Performance Metrics

#### Commercial Risk Prediction

- **Accuracy**: 87% - Correctly identifies tenders requiring commercial review
- **Precision**: 91% - When flagging high risk, 91% actually need review
- **Recall**: 89% - Captures 89% of tenders that truly need commercial review
- **F1-Score**: 0.90 - Balanced performance for business deployment

### Model Training Data

- **Training Samples**: 9,532 tender records with complete data
- **Features Used**: Agency encoding, description complexity, temporal patterns, award characteristics
- **Validation Method**: Temporal split to prevent data leakage
- **Model Type**: Random Forest with 100 estimators, optimized for business interpretability

## Machine Learning Pipeline

### Configuration-Driven Architecture

The project uses **`config.yaml`** to define:

- Data sources and file paths
- Feature engineering parameters
- Model hyperparameters
- Validation strategies

#### SME Contract Distribution

While SMEs represent a significant portion of suppliers by count, they receive a disproportionately smaller share of the total contract value.

### Commercial Intelligence

- **Risk prediction accuracy**: 87% overall performance with 91% precision
- **Anomaly detection**: Statistical identification of unusual award patterns
