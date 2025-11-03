"""
Prediction model for award amounts

This module implements a regression model to predict award amounts
from tender text and agency/category features.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
import joblib


class AwardAmountPredictor:
    """Predict award amounts from tender features"""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize predictor
        
        Args:
            model_type: 'random_forest', 'gradient_boosting', or 'ridge'
        """
        self.model_type = model_type
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:  # ridge
            self.model = Ridge(alpha=1.0, random_state=42)
        
        self.feature_importance = None
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"\n=== Training {self.model_type} model ===")
        self.model.fit(X_train, y_train)
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        print("Training complete")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print(f"\n=== Evaluating {self.model_type} model ===")
        
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Calculate metrics on original scale (exp transform)
        y_test_original = np.expm1(y_test)
        y_pred_original = np.expm1(y_pred)
        
        mae_original = mean_absolute_error(y_test_original, y_pred_original)
        print(f"\nMean Absolute Error (original scale): ${mae_original:,.2f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mae_original': mae_original
        }
    
    def save_model(self, filepath=None):
        """Save trained model"""
        if filepath is None:
            filepath = Path(__file__).parent.parent.parent / "data" / f"award_amount_predictor_{self.model_type}.pkl"
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")


def train_and_evaluate_models(preprocessor, df):
    """
    Train and evaluate multiple models for award amount prediction
    
    Returns:
        Dictionary of trained models
    """
    print("\n" + "="*60)
    print("AWARD AMOUNT PREDICTION")
    print("="*60)
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test = preprocessor.prepare_for_modeling(
        df, target='award_amount'
    )
    
    models = {}
    results = {}
    
    # Train different models
    for model_type in ['random_forest', 'gradient_boosting', 'ridge']:
        predictor = AwardAmountPredictor(model_type=model_type)
        predictor.train(X_train, y_train)
        metrics = predictor.evaluate(X_test, y_test)
        
        models[model_type] = predictor
        results[model_type] = metrics
        
        # Save model
        predictor.save_model()
    
    # Compare models
    print("\n=== Model Comparison ===")
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('r2', ascending=False)
    print(comparison_df)
    
    best_model = comparison_df.index[0]
    print(f"\nBest model: {best_model} (R² = {comparison_df.loc[best_model, 'r2']:.4f})")
    
    return models, results


if __name__ == "__main__":
    from preprocessing import preprocess_pipeline
    
    # Load and preprocess data
    preprocessor, df_processed = preprocess_pipeline()
    
    # Train and evaluate models
    models, results = train_and_evaluate_models(preprocessor, df_processed)
