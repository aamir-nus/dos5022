"""
Clustering module for agencies and suppliers

This module implements clustering to identify:
- Agency spending patterns
- Supplier profiles
- Category strategy insights
- Risk flags
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class SpendPatternClusterer:
    """Cluster agencies/suppliers by spending patterns"""
    
    def __init__(self, n_clusters=3):
        """
        Initialize clusterer
        
        Args:
            n_clusters: Number of clusters to create
        """
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = None
        
    def cluster_agencies(self, agency_features):
        """
        Cluster agencies by spending patterns
        
        Args:
            agency_features: DataFrame with agency-level aggregated features
        
        Returns:
            DataFrame with cluster assignments
        """
        print("\n=== Clustering Agencies ===")
        print(f"Number of agencies: {len(agency_features)}")
        
        # Select features for clustering
        feature_cols = ['total_spend', 'avg_spend', 'num_tenders', 
                       'num_categories', 'num_suppliers']
        
        X = agency_features[feature_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create new model instance for agencies
        agency_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        
        # Fit clustering model
        agency_model.fit(X_scaled)
        
        # Add cluster labels
        agency_features['cluster'] = agency_model.labels_
        
        # Analyze clusters
        print("\n=== Agency Cluster Analysis ===")
        for cluster_id in range(self.n_clusters):
            cluster_data = agency_features[agency_features['cluster'] == cluster_id]
            print(f"\nCluster {cluster_id} ({len(cluster_data)} agencies):")
            print(f"  Agencies: {', '.join(cluster_data['agency'].tolist())}")
            print(f"  Avg Total Spend: ${cluster_data['total_spend'].mean():,.2f}")
            print(f"  Avg # Tenders: {cluster_data['num_tenders'].mean():.1f}")
            print(f"  Avg # Categories: {cluster_data['num_categories'].mean():.1f}")
            print(f"  Avg # Suppliers: {cluster_data['num_suppliers'].mean():.1f}")
        
        return agency_features
    
    def cluster_suppliers(self, supplier_features):
        """
        Cluster suppliers by performance patterns
        
        Args:
            supplier_features: DataFrame with supplier-level aggregated features
        
        Returns:
            DataFrame with cluster assignments
        """
        print("\n=== Clustering Suppliers ===")
        print(f"Number of suppliers: {len(supplier_features)}")
        
        # Select features for clustering
        feature_cols = ['total_awards', 'avg_award', 'num_contracts',
                       'num_categories', 'num_agencies']
        
        X = supplier_features[feature_cols].fillna(0)
        
        # Scale features (use new scaler for suppliers)
        supplier_scaler = StandardScaler()
        X_scaled = supplier_scaler.fit_transform(X)
        
        # Create new model instance for suppliers
        supplier_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        
        # Fit clustering model
        supplier_model.fit(X_scaled)
        
        # Add cluster labels
        supplier_features['cluster'] = supplier_model.labels_
        
        # Analyze clusters
        print("\n=== Supplier Cluster Analysis ===")
        for cluster_id in range(self.n_clusters):
            cluster_data = supplier_features[supplier_features['cluster'] == cluster_id]
            print(f"\nCluster {cluster_id} ({len(cluster_data)} suppliers):")
            print(f"  Avg Total Awards: ${cluster_data['total_awards'].mean():,.2f}")
            print(f"  Avg Contract Value: ${cluster_data['avg_award'].mean():,.2f}")
            print(f"  Avg # Contracts: {cluster_data['num_contracts'].mean():.1f}")
            print(f"  Avg # Categories: {cluster_data['num_categories'].mean():.1f}")
            print(f"  Avg # Agencies: {cluster_data['num_agencies'].mean():.1f}")
        
        return supplier_features
    
    def identify_risk_flags(self, agency_features, supplier_features):
        """
        Identify potential risk flags based on clustering results
        
        Returns:
            Dictionary of risk insights
        """
        print("\n=== Identifying Risk Flags ===")
        
        risks = {
            'high_concentration_agencies': [],
            'high_concentration_suppliers': [],
            'low_diversity_agencies': [],
            'single_category_suppliers': []
        }
        
        # Agency risks: high concentration with few suppliers
        for _, row in agency_features.iterrows():
            if row['num_tenders'] > 20 and row['num_suppliers'] < 5:
                risks['high_concentration_agencies'].append({
                    'agency': row['agency'],
                    'num_tenders': row['num_tenders'],
                    'num_suppliers': row['num_suppliers'],
                    'risk': 'Low supplier diversity - dependency risk'
                })
        
        # Agency risks: low category diversity
        for _, row in agency_features.iterrows():
            if row['num_tenders'] > 15 and row['num_categories'] < 3:
                risks['low_diversity_agencies'].append({
                    'agency': row['agency'],
                    'num_tenders': row['num_tenders'],
                    'num_categories': row['num_categories'],
                    'risk': 'Low category diversity - may miss opportunities'
                })
        
        # Supplier risks: high concentration with single client
        for _, row in supplier_features.iterrows():
            if row['num_contracts'] > 10 and row['num_agencies'] == 1:
                risks['high_concentration_suppliers'].append({
                    'supplier': row['supplier'],
                    'num_contracts': row['num_contracts'],
                    'num_agencies': row['num_agencies'],
                    'risk': 'Single agency dependency - business risk'
                })
        
        # Supplier risks: single category specialization
        for _, row in supplier_features.iterrows():
            if row['num_contracts'] > 5 and row['num_categories'] == 1:
                risks['single_category_suppliers'].append({
                    'supplier': row['supplier'],
                    'num_contracts': row['num_contracts'],
                    'num_categories': row['num_categories'],
                    'risk': 'Single category - limited capability breadth'
                })
        
        # Print risk summary
        print(f"\nRisk Summary:")
        print(f"  Agencies with low supplier diversity: {len(risks['high_concentration_agencies'])}")
        print(f"  Agencies with low category diversity: {len(risks['low_diversity_agencies'])}")
        print(f"  Suppliers with single agency dependency: {len(risks['high_concentration_suppliers'])}")
        print(f"  Suppliers with single category: {len(risks['single_category_suppliers'])}")
        
        return risks
    
    def visualize_clusters(self, features, entity_col, output_dir=None):
        """
        Visualize clustering results
        
        Args:
            features: DataFrame with cluster assignments
            entity_col: Column name for entity (agency or supplier)
            output_dir: Directory to save plots
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "data" / "visualizations"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Cluster distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count by cluster
        cluster_counts = features['cluster'].value_counts().sort_index()
        axes[0].bar(cluster_counts.index, cluster_counts.values, color=['red', 'green', 'blue'])
        axes[0].set_xlabel('Cluster')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'{entity_col.capitalize()} Distribution by Cluster')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Average metrics by cluster
        if 'total_spend' in features.columns:
            metric = 'total_spend'
            ylabel = 'Total Spend'
        else:
            metric = 'total_awards'
            ylabel = 'Total Awards'
        
        cluster_avg = features.groupby('cluster')[metric].mean()
        axes[1].bar(cluster_avg.index, cluster_avg.values, color=['red', 'green', 'blue'])
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel(f'Average {ylabel}')
        axes[1].set_title(f'Average {ylabel} by Cluster')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{entity_col}_clusters.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / f'{entity_col}_clusters.png'}")
        plt.close()


def perform_clustering_analysis(preprocessor, df):
    """
    Perform complete clustering analysis
    
    Returns:
        Clustered agency and supplier features, plus risk flags
    """
    print("\n" + "="*60)
    print("CLUSTERING ANALYSIS")
    print("="*60)
    
    # Get clustering features
    agency_features, supplier_features = preprocessor.get_clustering_features(df)
    
    # Initialize clusterer
    clusterer = SpendPatternClusterer(n_clusters=3)
    
    # Cluster agencies
    agency_features_clustered = clusterer.cluster_agencies(agency_features)
    clusterer.visualize_clusters(agency_features_clustered, 'agency')
    
    # Cluster suppliers
    supplier_features_clustered = clusterer.cluster_suppliers(supplier_features)
    clusterer.visualize_clusters(supplier_features_clustered, 'supplier')
    
    # Identify risk flags
    risks = clusterer.identify_risk_flags(agency_features_clustered, supplier_features_clustered)
    
    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data"
    agency_features_clustered.to_csv(output_dir / "agency_clusters.csv", index=False)
    supplier_features_clustered.to_csv(output_dir / "supplier_clusters.csv", index=False)
    
    # Save risk report
    risk_report = []
    for risk_type, risk_list in risks.items():
        for risk_item in risk_list:
            risk_report.append({
                'risk_type': risk_type,
                **risk_item
            })
    
    if risk_report:
        risk_df = pd.DataFrame(risk_report)
        risk_df.to_csv(output_dir / "risk_flags.csv", index=False)
        print(f"\nRisk report saved to {output_dir / 'risk_flags.csv'}")
    
    return agency_features_clustered, supplier_features_clustered, risks


if __name__ == "__main__":
    from preprocessing import preprocess_pipeline
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    # Load and preprocess data
    preprocessor, df_processed = preprocess_pipeline()
    
    # Perform clustering analysis
    agency_clusters, supplier_clusters, risks = perform_clustering_analysis(
        preprocessor, df_processed
    )
