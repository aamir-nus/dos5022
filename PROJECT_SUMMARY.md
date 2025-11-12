# DOS5022 Tender Analysis Project - Summary

## Repository Structure

- Created on dev branch (working on feature branch)
- `src/models` directory structure implemented
- Using `uv` with Python 3.10 (NOT pip)

### Data Acquisition - General Stats

- **Source**: https://data.gov.sg/datasets/d_acde1106003906a75c3fa052592f2fcb/view
- **Dataset Size**: 18,021 records, cleaned to 11,915 unique tenders
- **Scope**: 111 government agencies, 4,134 unique suppliers
- **Total Value**: S$102 billion in awarded contracts
- **Award Success Rate**: 94.1%

### Machine Learning Capabilities

#### 1. Supplier Portfolio Analysis

Strategic supplier segmentation using K-Means clustering, then analyzing each cluster and assigning appropriate names. Subjective judgement was used to combine into three segements instead of 5/4 clusters:

- **5 clusters -> whittled to 4 clusters**: Based on total awards, agency diversity, contract consistency
- **Key Insights**:

  - 4,133 active suppliers analyzed
  - Market concentration: 17.7% controlled by top 10 suppliers

#### 2. Predictive Commercial Risk Assessment

Random Forest classifier for unusual award detection:

- **Objective**: Flag tenders requiring commercial review
- **Model**: Constrained Random Forest (overfitting prevention)
- **Performance**: 86.8% test accuracy with temporal validation
- **Risk Distribution**: 3.4% high-risk tenders, 96.6% normal tenders

#### 3. Advanced Clustering Analysis

Comprehensive clustering evaluation:

- **Supplier Clustering**: Hierarchical performed best (Silhouette: 0.763), *K-Means chosen for complete coverage*
- ~~**Agency Clustering**: K-Means performed best (Silhouette: 0.518)~~
- **DBSCAN**: Limited success due to data characteristics
- **Model Selection**: *K-Means preferred for complete cluster assignment*

## Technical Implementation

### Project Structure

```
dos5022/
├── src/
│   ├── download_data.py          # Data acquisition
│   ├── preprocessing.py          # Feature engineering & data preparation
│   ├── visualization.py          # EDA visualizations
│   └── models/
│       ├── award_classifier.py   # Award status classification
│       └── clustering.py         # Clustering & risk detection
├── notebooks/                    # Analysis notebooks
│   ├── 01_preprocessing.ipynb    # Data preprocessing pipeline
│   ├── 02_modeling.ipynb         # Clustering analysis
│   └── supplier clustering and risk prediction.ipynb  # Business analysis
├── main.py                       # Complete pipeline
├── pyproject.toml               # uv dependencies
├── config.yaml                  # Configuration-driven approach
├── README.md                    # Main documentation
├── USAGE.md                     # Usage guide
└── PROJECT_SUMMARY.md           # This file
```

### Project Features

- **Configuration-driven**: YAML-based configuration for reproducibility
- **Temporal Validation**: Prevents data leakage in time-series data
- **Overfitting Prevention**: Constrained model complexity and validation strategies
- **Memory Optimization**: Efficient processing of large datasets (18,021 records)

### Dependencies

- **NOTE : project uses `uv` not `pip`**
- create environment and install packages : `uv sync --all-groups`
- activate virtual environment : `source .venv/bin/activate`
- add new packages : `uv add <package name>`

### Outputs (Comprehensive)

1. **Data Files**:

   - Processed tender data (CSV) - 11,915 records, 24 features
   - Agency clustering features (CSV) - 110 agencies
   - Supplier clustering features (CSV) - 4,133 suppliers
   - Original dataset (Excel) - 18,021 records
2. **Models**:

   - Supplier clustering model (PKL) - K-Means with scaler
   - Commercial risk predictor (PKL) - Random Forest classifier
   - Clustering results (CSV) - Agency and supplier assignments
   - Model metadata (JSON) - Performance metrics and feature importance
3. **Analysis Results**:

   - Business analysis summary (JSON) - Complete strategic insights
   - Model performance metrics
   - Risk distribution analysis
   - Supplier portfolio insights

## Usage

### Jupyter Notebooks (Analysis)

```bash
# Complete analysis pipeline
jupyter notebooks/01_preprocessing.ipynb
jupyter notebooks/02_modeling.ipynb
jupyter notebooks/supplier clustering and risk prediction.ipynb
```

## Business Intelligence Results

### Dataset Overview

- **11,915 tender records** (cleaned from 18,021 original records)
- **111 government agencies**, **4,134 unique suppliers**
- **Total Contract Value**: S$102 billion across awarded tenders
- **Time Period**: 2020-2025 with temporal validation splits
- **Award Success Rate**: 94.1%

### Supplier Portfolio Analysis

**Strategic Segmentation - based on Subjective Analysis of Clusters:**

- **Large Scale High Value**: 1.4% of suppliers controlling 46.5% of market value
- **Medium Scale Specialized**: 87.5% of suppliers in specialized roles (35.2% market value)
- **Small Scale Broad Reach**: 11.2% of suppliers with agency relationships (18.2% market value)

**Market Health Indicators:**

- **Market Concentration**: Moderate (17.7% controlled by top 10 suppliers)
- **Dependency Risk**: 66.7% suppliers belong to the same agency (monitoring recommended)

### Commercial Risk Assessment

**Model Performance:**

- **Accuracy**: 86.8% on temporal validation (2020-2024 train, 2025 test)
- **Risk Detection Logic**: Predict risk -> flag unusual awards (>2σ from agency means)

**Risk Distribution:**

- **High Risk Tenders**: 3.4% (384 out of 11,216 awarded tenders)
- **Normal Risk Tenders**: 96.6%
- **Class Imbalance**: Addressed with balanced class weights

### Clustering Analysis

**Supplier Clustering (Preferred: K-Means):**

- **Silhouette Score**: 0.732 (excellent separation)
- **Optimal Clusters**: clusters combined into 3 segments for strategic portfolio management
- **Business Value**: Clear strategic segments for supplier relationship management

## Strategic Business Insights

### Supplier Relationship Management

1. **Strategic Partnerships**: Focus on Large Scale High Value suppliers (1.4% controlling 46.5% of market)
2. **SME Development**: Target Small Scale suppliers for capability building programs
3. **Niche Protection**: Protect Medium Scale Specialized suppliers for category expertise
4. **Diversification**: Encourage multi-agency supplier relationships to reduce dependency

### Data-Driven Procurement Strategy

1. **Risk Management**: Proactive identification of unusual award patterns
2. **Performance Monitoring**: Supplier segment performance tracking and evaluation
