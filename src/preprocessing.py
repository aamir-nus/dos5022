"""
Data preprocessing module for Singapore Government Tender Analysis

This module handles:
- Data loading and cleaning
- Feature engineering based on actual dataset structure
- Categorical encoding and scaling
- Train-test splitting
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class TenderDataPreprocessor:
    """
    Preprocessor for Singapore Government Tender data
    Steps:
        1. Load data from CSV
        2. Clean data (handle missing values, duplicates, data types)
        3. Engineer features based on actual dataset structure
        4. Encode categorical variables
        5. Prepare data for modeling (train-test split, scaling)

    Args:
        config_path: Path to YAML configuration file

    Attributes:
        config: Loaded configuration dictionary
        label_encoders: Dictionary of fitted LabelEncoders for categorical features
        scaler: Fitted StandardScaler for numerical features
        data_config: Data-related configuration
        feature_config: Feature engineering configuration
        preprocessing_config: Preprocessing configuration
    """

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.data_config = self.config['data']
        self.feature_config = self.config['features']
        self.preprocessing_config = self.config['preprocessing']

    def load_data(self, filepath=None, nrows=None):
        """
        Load tender data from CSV

        Args:
            filepath: Path to CSV file
            nrows: Number of rows to read (for memory efficiency)

        Returns:
            DataFrame with tender data
        """
        if filepath is None:
            filepath = self.data_config['source_file']

        # Resolve path relative to project root
        if not Path(filepath).is_absolute():
            # Get the project root (parent of src directory)
            project_root = Path(__file__).parent.parent
            filepath = project_root / filepath

        #check extension of the file
        df = pd.DataFrame()
        if str(filepath).split(".")[-1] == "csv":
            df = pd.read_csv(filepath, nrows=nrows)
        if str(filepath).split(".")[-1] == "xlsx":
            df = pd.read_excel(filepath, nrows=nrows)

        print(f"Loaded {len(df)} records from {filepath}")
        return df
    
    def clean_data(self, df):
        """
        Clean the data - handle missing values, duplicates, data types

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        print("\n=== Cleaning Data ===")

        # Create a copy to avoid modifying original
        df_clean = df.copy()

        # Check for duplicates
        tender_no_col = self.data_config['columns']['tender_no']
        duplicates = df_clean.duplicated(subset=[tender_no_col]).sum()
        if duplicates > 0:
            print(f"Removing {duplicates} duplicate records")
            df_clean = df_clean.drop_duplicates(subset=[tender_no_col])

        # Convert award_date to datetime
        award_date_col = self.data_config['columns']['award_date']
        df_clean[award_date_col] = pd.to_datetime(df_clean[award_date_col], errors='coerce')

        # Handle missing values based on config
        awarded_amt_col = self.data_config['columns']['awarded_amt']
        supplier_col = self.data_config['columns']['supplier_name']

        # Fill missing award amounts with 0
        if awarded_amt_col in df_clean.columns:
            df_clean[awarded_amt_col] = df_clean[awarded_amt_col].fillna(
                self.preprocessing_config['missing_values']['awarded_amt']
            )

        # Fill missing supplier names
        if supplier_col in df_clean.columns:
            df_clean[supplier_col] = df_clean[supplier_col].fillna(
                self.preprocessing_config['missing_values']['supplier_name']
            )

        print(f"Clean data shape: {df_clean.shape}")
        print(f"Missing values:\n{df_clean.isnull().sum()}")

        return df_clean
    
    def engineer_features(self, df):
        """
        Create new features from existing data based on config

        Args:
            df: Cleaned DataFrame

        Returns:
            DataFrame with engineered features
        """
        print("\n=== Engineering Features ===")

        df_feat = df.copy()

        # Get column names from config
        tender_desc_col = self.data_config['columns']['tender_description']
        award_date_col = self.data_config['columns']['award_date']
        awarded_amt_col = self.data_config['columns']['awarded_amt']
        status_col = self.data_config['columns']['tender_detail_status']

        # Extract date features from config
        date_features = self.feature_config['date_features']
        if 'award_year' in date_features:
            df_feat['award_year'] = df_feat[award_date_col].dt.year
        if 'award_month' in date_features:
            df_feat['award_month'] = df_feat[award_date_col].dt.month
        if 'award_quarter' in date_features:
            df_feat['award_quarter'] = df_feat[award_date_col].dt.quarter
        if 'award_day_of_week' in date_features:
            df_feat['award_day_of_week'] = df_feat[award_date_col].dt.dayofweek

        # Text length features from config
        if 'description_length' in self.feature_config['text_features']:
            df_feat['description_length'] = df_feat[tender_desc_col].str.len()
        if 'description_word_count' in self.feature_config['text_features']:
            df_feat['description_word_count'] = df_feat[tender_desc_col].str.split().str.len()
        if 'description_char_count' in self.feature_config['text_features']:
            df_feat['description_char_count'] = df_feat[tender_desc_col].str.len()

        # Binary features from config
        if 'is_awarded' in self.feature_config['derived_features']:
            df_feat['is_awarded'] = (df_feat[awarded_amt_col] > 0).astype(int)

        # Log transform of award amount from config
        if 'log_awarded_amt' in self.feature_config['derived_features']:
            df_feat['log_awarded_amt'] = np.log1p(df_feat[awarded_amt_col])

        # Award amount categories (for classification)
        if 'award_amount_category' in self.feature_config['derived_features']:
            df_feat['award_amount_category'] = pd.cut(
                df_feat[awarded_amt_col],
                bins=[-1, 0, 10000, 50000, float('inf')],
                labels=['No Award', 'Small', 'Medium', 'Large']
            ).astype(str)

        print(f"Added features. New shape: {df_feat.shape}")
        new_features = [col for col in df_feat.columns if col not in df.columns]
        print(f"New features: {new_features}")

        return df_feat
    
    def encode_categorical(self, df, columns=None):
        """
        Encode categorical variables based on config

        Args:
            df: DataFrame with engineered features
            columns: List of columns to encode (uses config if None)

        Returns:
            DataFrame with encoded categorical features
        """
        print("\n=== Encoding Categorical Features ===")

        if columns is None:
            columns = self.feature_config['categorical_features']

        df_encoded = df.copy()

        for col in columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                # Handle unseen values during encoding
                df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded {col}: {len(le.classes_)} unique values")
            else:
                print(f"Warning: Column {col} not found in data")

        return df_encoded
    
    def prepare_for_modeling(self, df, target_type='regression', test_size=0.2, random_state=None):
        """
        Prepare data for modeling based on config

        Args:
            df: DataFrame with encoded features
            target_type: Type of modeling ('regression', 'classification', 'clustering')
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility

        Returns:
            For regression/classification: X_train, X_test, y_train, y_test
            For clustering: feature DataFrame
        """
        print(f"\n=== Preparing Data for {target_type.title()} Modeling ===")

        if random_state is None:
            random_state = self.config['random_state']

        # Determine target based on modeling type
        if target_type == 'regression':
            target = self.config['targets']['regression']['primary']
            # Filter to only awarded tenders for amount prediction
            df_model = df[df['is_awarded'] == 1].copy()
            print(f"Using {len(df_model)} awarded tenders for amount prediction")
        elif target_type == 'classification':
            target = self.config['targets']['classification']['primary']
            df_model = df.copy()
        else:  # clustering
            return self.prepare_for_clustering(df)

        # Select available features from config
        feature_cols = []
        categorical_cols = self.feature_config['categorical_features']
        numerical_cols = self.feature_config['numerical_features']

        # Add encoded categorical features
        for col in categorical_cols:
            encoded_col = f'{col}_encoded'
            if encoded_col in df_model.columns:
                feature_cols.append(encoded_col)

        # Add numerical features
        for col in numerical_cols:
            if col in df_model.columns and col != target:
                feature_cols.append(col)

        print(f"Using features: {feature_cols}")

        # Prepare features and target
        X = df_model[feature_cols]
        y = df_model[target]

        # Handle missing values in features
        X = X.fillna(0)

        if target_type != 'clustering':
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

    def prepare_for_clustering(self, df):
        """
        Prepare data for clustering analysis

        Args:
            df: DataFrame with encoded features

        Returns:
            Tuple of (agency_features, supplier_features) DataFrames
        """
        print("\n=== Preparing Data for Clustering ===")

        # Get column names from config
        agency_col = self.data_config['columns']['agency']
        supplier_col = self.data_config['columns']['supplier_name']
        awarded_amt_col = self.data_config['columns']['awarded_amt']
        category_col = self.data_config['columns']['tender_description']  # Using description as category proxy

        # Filter to awarded tenders
        awarded_df = df[df['is_awarded'] == 1].copy()

        # Agency-level aggregation
        agency_features = awarded_df.groupby(agency_col).agg({
            awarded_amt_col: ['sum', 'mean', 'count', 'std'],
            supplier_col: lambda x: x.nunique(),  # Number of unique suppliers
        }).reset_index()

        agency_features.columns = ['agency', 'total_spend', 'avg_spend', 'num_tenders',
                                   'std_spend', 'num_suppliers']

        # Supplier-level aggregation
        supplier_features = awarded_df.groupby(supplier_col).agg({
            awarded_amt_col: ['sum', 'mean', 'count', 'std'],
            agency_col: lambda x: x.nunique(),     # Number of unique agencies
        }).reset_index()

        supplier_features.columns = ['supplier', 'total_awards', 'avg_award', 'num_contracts',
                                     'std_awards', 'num_agencies']

        print(f"Agency features shape: {agency_features.shape}")
        print(f"Supplier features shape: {supplier_features.shape}")

        return agency_features, supplier_features
    
    def save_processed_data(self, df, output_path=None):
        """
        Save processed data to file

        Args:
            df: Processed DataFrame
            output_path: Path to save file (uses config if None)
        """
        if output_path is None:
            output_path = self.data_config['processed_file']

        # Resolve path relative to project root
        if not Path(output_path).is_absolute():
            # Get the project root (parent of src directory)
            project_root = Path(__file__).parent.parent
            output_path = project_root / output_path

        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")


def preprocess_pipeline(config_path=None, nrows=None):
    """
    Complete preprocessing pipeline using config

    Args:
        config_path: Path to config file
        nrows: Number of rows to read from source (for memory efficiency)

    Returns:
        Tuple of (preprocessor object, processed dataframe)
    """
    preprocessor = TenderDataPreprocessor(config_path)

    # Load data
    df = preprocessor.load_data(nrows=nrows)

    # Clean data
    df = preprocessor.clean_data(df)

    # Engineer features
    df = preprocessor.engineer_features(df)

    # Encode categorical features
    df = preprocessor.encode_categorical(df)

    # Save processed data
    preprocessor.save_processed_data(df)

    return preprocessor, df


if __name__ == "__main__":
    # Run preprocessing pipeline
    print("Starting preprocessing pipeline...")
    preprocessor, df_processed = preprocess_pipeline()

    print("\n=== Preprocessing Complete ===")
    print(f"Final data shape: {df_processed.shape}")
    print(f"\nSample of processed data:")
    print(df_processed.head())

    # Print column information
    print(f"\nProcessed columns:")
    for i, col in enumerate(df_processed.columns, 1):
        print(f"{i:2d}. {col}")
