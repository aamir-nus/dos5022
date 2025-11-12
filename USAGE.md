# Usage Guide

## Quick Start

```bash
# Run the complete pipeline
uv run python main.py
```

This will execute all steps and generate outputs in the `data/` directory.

## Individual Components

### 1. Data Acquisition

```bash
uv run python src/download_data.py
```

Downloads tender data from data.gov.sg or generates sample data if network is restricted.

### 2. Data Preprocessing

```bash
uv run python src/preprocessing.py
```

Cleans data, engineers features, and encodes categorical variables.

### 3. Visualizations

```bash
uv run python src/visualization.py
```

Generates exploratory data analysis plots:

- Award distribution
- Award amounts distribution
- Agency clusters
- Supplier clusters

### 4. Award Amount Prediction

```bash
uv run python src/models/award_predictor.py
```

Trains 3 regression models to predict tender award amounts:

- Random Forest
- Gradient Boosting
- Ridge Regression

### 5. Award Status Classification

```bash
uv run python src/models/award_classifier.py
```

Trains 3 classification models to predict "Awarded" vs "No award":

- Random Forest Classifier
- Gradient Boosting Classifier
- Logistic Regression

### 6. Clustering Analysis

```bash
uv run python src/models/clustering.py
```

Performs K-Means clustering on:

- Agencies (by spending patterns)
- Suppliers (by performance patterns)

Identifies risk flags:

- Low supplier diversity
- Low category diversity
- Single agency dependency
- Single category focus

## Output Files

All outputs are saved in `data/`:

### Data Files

- `tender_data.csv` - Raw tender data
- `tender_data_processed.csv` - Processed data with engineered features
- `agency_clusters.csv` - Agency clustering results
- `supplier_clusters.csv` - Supplier clustering results
- `risk_flags.csv` - Identified risk flags (if any)

### Model Files

- `award_status_classifier_*.pkl` - Trained classification models

### Visualizations

- `visualizations/award_distribution.png`
- `visualizations/award_amounts_distribution.png`
- `visualizations/agency_analysis.png`
- `visualizations/category_analysis.png`
- `visualizations/temporal_trends.png`
- `visualizations/correlation_heatmap.png`
- `visualizations/agency_clusters.png`
- `visualizations/supplier_clusters.png`

## Python API

You can also use the modules programmatically:

```python
from src.download_data import generate_sample_data
from src.preprocessing import preprocess_pipeline
from src.visualization import TenderVisualizer
from src.models.award_predictor import train_and_evaluate_models
from src.models.award_classifier import train_and_evaluate_classifiers
from src.models.clustering import perform_clustering_analysis

# Generate sample data
df = generate_sample_data(n_samples=1000)

# Preprocess
preprocessor, df_processed = preprocess_pipeline()

# Visualize
visualizer = TenderVisualizer()
visualizer.generate_all_visualizations(df_processed)

# Train models
models, pred_results = train_and_evaluate_models(preprocessor, df_processed)
classifiers, class_results = train_and_evaluate_classifiers(preprocessor, df_processed)

# Clustering
agency_clusters, supplier_clusters, risks = perform_clustering_analysis(preprocessor, df_processed)
```
