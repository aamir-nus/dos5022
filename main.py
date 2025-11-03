"""
Main execution script for tender analysis pipeline

This script demonstrates the complete workflow:
1. Data download/generation
2. Preprocessing
3. Visualization
4. Award amount prediction
5. Award status classification
6. Agency/supplier clustering
"""
from pathlib import Path
import sys

# Ensure src is in path for imports
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import modules
from download_data import download_dataset
from preprocessing import preprocess_pipeline
from visualization import TenderVisualizer
from models.award_predictor import train_and_evaluate_models
from models.award_classifier import train_and_evaluate_classifiers
from models.clustering import perform_clustering_analysis


def main():
    """Run complete tender analysis pipeline"""
    
    print("="*80)
    print(" TENDER ANALYSIS PIPELINE ".center(80))
    print("="*80)
    
    # Step 1: Download/generate data
    print("\n" + "="*80)
    print("STEP 1: DATA ACQUISITION")
    print("="*80)
    df = download_dataset()
    
    if df is None:
        print("Error: Failed to download data")
        return
    
    # Step 2: Preprocess data
    print("\n" + "="*80)
    print("STEP 2: DATA PREPROCESSING")
    print("="*80)
    preprocessor, df_processed = preprocess_pipeline()
    
    # Step 3: Generate visualizations
    print("\n" + "="*80)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("="*80)
    visualizer = TenderVisualizer()
    visualizer.generate_all_visualizations(df_processed)
    
    # Step 4: Train award amount prediction models
    print("\n" + "="*80)
    print("STEP 4: AWARD AMOUNT PREDICTION")
    print("="*80)
    models, pred_results = train_and_evaluate_models(preprocessor, df_processed)
    
    # Step 5: Train award status classification models
    print("\n" + "="*80)
    print("STEP 5: AWARD STATUS CLASSIFICATION")
    print("="*80)
    classifiers, class_results = train_and_evaluate_classifiers(preprocessor, df_processed)
    
    # Step 6: Perform clustering analysis
    print("\n" + "="*80)
    print("STEP 6: CLUSTERING ANALYSIS")
    print("="*80)
    agency_clusters, supplier_clusters, risks = perform_clustering_analysis(
        preprocessor, df_processed
    )
    
    # Final summary
    print("\n" + "="*80)
    print(" PIPELINE COMPLETE ".center(80))
    print("="*80)
    print("\nSummary:")
    print(f"  - Processed {len(df_processed)} tender records")
    print(f"  - Generated {6} visualizations")
    print(f"  - Trained {len(models)} prediction models")
    print(f"  - Trained {len(classifiers)} classification models")
    print(f"  - Clustered {len(agency_clusters)} agencies")
    print(f"  - Clustered {len(supplier_clusters)} suppliers")
    print(f"  - Identified {sum(len(v) for v in risks.values())} risk flags")
    
    print("\nOutput files saved to: data/")
    print("  - tender_data.csv (raw data)")
    print("  - tender_data_processed.csv (processed data)")
    print("  - visualizations/*.png (EDA plots)")
    print("  - award_amount_predictor_*.pkl (prediction models)")
    print("  - award_status_classifier_*.pkl (classification models)")
    print("  - agency_clusters.csv (agency clustering results)")
    print("  - supplier_clusters.csv (supplier clustering results)")
    print("  - risk_flags.csv (identified risks)")
    
    print("\n" + "="*80)
    
    return {
        'preprocessor': preprocessor,
        'df_processed': df_processed,
        'models': models,
        'classifiers': classifiers,
        'agency_clusters': agency_clusters,
        'supplier_clusters': supplier_clusters,
        'risks': risks
    }


if __name__ == "__main__":
    results = main()

