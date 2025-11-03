"""
Classification model for award status

This module implements a classifier to predict likelihood of
"Awarded" vs "No award" status.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from pathlib import Path
import joblib


class AwardStatusClassifier:
    """Classify tenders as 'Awarded' or 'No award'"""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize classifier
        
        Args:
            model_type: 'random_forest', 'gradient_boosting', or 'logistic'
        """
        self.model_type = model_type
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:  # logistic
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        
        self.feature_importance = None
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"\n=== Training {self.model_type} classifier ===")
        self.model.fit(X_train, y_train)
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        print("Training complete")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print(f"\n=== Evaluating {self.model_type} classifier ===")
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['No award', 'Awarded']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"                Predicted No award  Predicted Awarded")
        print(f"Actual No award        {cm[0][0]:5d}              {cm[0][1]:5d}")
        print(f"Actual Awarded         {cm[1][0]:5d}              {cm[1][1]:5d}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
    
    def save_model(self, filepath=None):
        """Save trained model"""
        if filepath is None:
            filepath = Path(__file__).parent.parent.parent / "data" / f"award_status_classifier_{self.model_type}.pkl"
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")


def train_and_evaluate_classifiers(preprocessor, df):
    """
    Train and evaluate multiple classifiers for award status prediction
    
    Returns:
        Dictionary of trained classifiers
    """
    print("\n" + "="*60)
    print("AWARD STATUS CLASSIFICATION")
    print("="*60)
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test = preprocessor.prepare_for_modeling(
        df, target='is_awarded'
    )
    
    classifiers = {}
    results = {}
    
    # Train different classifiers
    for model_type in ['random_forest', 'gradient_boosting', 'logistic']:
        classifier = AwardStatusClassifier(model_type=model_type)
        classifier.train(X_train, y_train)
        metrics = classifier.evaluate(X_test, y_test)
        
        classifiers[model_type] = classifier
        results[model_type] = metrics
        
        # Save model
        classifier.save_model()
    
    # Compare models
    print("\n=== Model Comparison ===")
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('f1', ascending=False)
    print(comparison_df)
    
    best_model = comparison_df.index[0]
    print(f"\nBest model: {best_model} (F1 = {comparison_df.loc[best_model, 'f1']:.4f})")
    
    return classifiers, results


if __name__ == "__main__":
    from preprocessing import preprocess_pipeline
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    # Load and preprocess data
    preprocessor, df_processed = preprocess_pipeline()
    
    # Train and evaluate classifiers
    classifiers, results = train_and_evaluate_classifiers(preprocessor, df_processed)
