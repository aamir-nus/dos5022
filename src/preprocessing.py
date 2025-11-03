"""
Data preprocessing module for tender analysis

This module handles:
- Data cleaning
- Feature engineering
- Text preprocessing
- Train-test splitting
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path


class TenderDataPreprocessor:
    """Preprocessor for tender data"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath=None):
        """Load tender data from CSV"""
        if filepath is None:
            filepath = Path(__file__).parent.parent / "data" / "tender_data.csv"
        
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records")
        return df
    
    def clean_data(self, df):
        """Clean the data - handle missing values, duplicates, etc."""
        print("\n=== Cleaning Data ===")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Check for duplicates
        duplicates = df_clean.duplicated(subset=['tender_id']).sum()
        if duplicates > 0:
            print(f"Removing {duplicates} duplicate records")
            df_clean = df_clean.drop_duplicates(subset=['tender_id'])
        
        # Convert dates to datetime
        df_clean['published_date'] = pd.to_datetime(df_clean['published_date'])
        df_clean['award_date'] = pd.to_datetime(df_clean['award_date'], errors='coerce')
        
        # Fill missing award amounts with 0 for "No award" cases
        df_clean.loc[df_clean['award_status'] == 'No award', 'award_amount'] = 0
        
        print(f"Clean data shape: {df_clean.shape}")
        print(f"Missing values:\n{df_clean.isnull().sum()}")
        
        return df_clean
    
    def engineer_features(self, df):
        """Create new features from existing data"""
        print("\n=== Engineering Features ===")
        
        df_feat = df.copy()
        
        # Extract date features
        df_feat['published_year'] = df_feat['published_date'].dt.year
        df_feat['published_month'] = df_feat['published_date'].dt.month
        df_feat['published_quarter'] = df_feat['published_date'].dt.quarter
        
        # Text length features
        df_feat['description_length'] = df_feat['tender_description'].str.len()
        df_feat['description_word_count'] = df_feat['tender_description'].str.split().str.len()
        
        # Award status binary
        df_feat['is_awarded'] = (df_feat['award_status'] == 'Awarded').astype(int)
        
        # Days to award (for awarded tenders)
        df_feat['days_to_award'] = (df_feat['award_date'] - df_feat['published_date']).dt.days
        
        # Log transform of award amount (for awarded tenders)
        df_feat['log_award_amount'] = np.log1p(df_feat['award_amount'].fillna(0))
        
        print(f"Added features. New shape: {df_feat.shape}")
        print(f"New columns: {df_feat.columns.tolist()[-8:]}")
        
        return df_feat
    
    def encode_categorical(self, df, columns=['agency', 'category']):
        """Encode categorical variables"""
        print("\n=== Encoding Categorical Features ===")
        
        df_encoded = df.copy()
        
        for col in columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
                print(f"Encoded {col}: {len(le.classes_)} unique values")
        
        return df_encoded
    
    def prepare_for_modeling(self, df, target='award_amount', test_size=0.2, random_state=42):
        """
        Prepare data for modeling
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        print("\n=== Preparing Data for Modeling ===")
        
        # Select features for modeling
        feature_cols = [
            'agency_encoded', 'category_encoded',
            'published_year', 'published_month', 'published_quarter',
            'description_length', 'description_word_count'
        ]
        
        # Filter to only awarded tenders if predicting amount
        if target == 'award_amount':
            df_model = df[df['is_awarded'] == 1].copy()
            print(f"Using {len(df_model)} awarded tenders for amount prediction")
        else:
            df_model = df.copy()
        
        # Prepare features and target
        X = df_model[feature_cols]
        
        if target == 'award_amount':
            y = df_model['log_award_amount']
        elif target == 'is_awarded':
            y = df_model['is_awarded']
        else:
            y = df_model[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_clustering_features(self, df):
        """
        Prepare features for clustering agencies/suppliers by spend patterns
        
        Returns:
            DataFrame with aggregated spend features
        """
        print("\n=== Preparing Clustering Features ===")
        
        # Filter to awarded tenders
        awarded_df = df[df['is_awarded'] == 1].copy()
        
        # Agency-level aggregation
        agency_features = awarded_df.groupby('agency').agg({
            'award_amount': ['sum', 'mean', 'count', 'std'],
            'category': lambda x: x.nunique(),  # Number of unique categories
            'supplier': lambda x: x.nunique()   # Number of unique suppliers
        }).reset_index()
        
        agency_features.columns = ['agency', 'total_spend', 'avg_spend', 'num_tenders', 
                                   'std_spend', 'num_categories', 'num_suppliers']
        
        # Supplier-level aggregation
        supplier_features = awarded_df.groupby('supplier').agg({
            'award_amount': ['sum', 'mean', 'count', 'std'],
            'category': lambda x: x.nunique(),  # Number of unique categories
            'agency': lambda x: x.nunique()     # Number of unique agencies
        }).reset_index()
        
        supplier_features.columns = ['supplier', 'total_awards', 'avg_award', 'num_contracts',
                                     'std_awards', 'num_categories', 'num_agencies']
        
        print(f"Agency features shape: {agency_features.shape}")
        print(f"Supplier features shape: {supplier_features.shape}")
        
        return agency_features, supplier_features


def preprocess_pipeline(data_path=None):
    """
    Complete preprocessing pipeline
    
    Returns:
        Preprocessor object and cleaned/engineered dataframe
    """
    preprocessor = TenderDataPreprocessor()
    
    # Load data
    df = preprocessor.load_data(data_path)
    
    # Clean data
    df = preprocessor.clean_data(df)
    
    # Engineer features
    df = preprocessor.engineer_features(df)
    
    # Encode categorical features
    df = preprocessor.encode_categorical(df)
    
    return preprocessor, df


if __name__ == "__main__":
    # Run preprocessing pipeline
    preprocessor, df_processed = preprocess_pipeline()
    
    print("\n=== Preprocessing Complete ===")
    print(f"Final data shape: {df_processed.shape}")
    print(f"\nSample of processed data:")
    print(df_processed.head())
    
    # Save processed data
    output_path = Path(__file__).parent.parent / "data" / "tender_data_processed.csv"
    df_processed.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to {output_path}")
