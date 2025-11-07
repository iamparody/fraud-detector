# prediction/model_predictor.py
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Union
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudPredictor:
    def __init__(self, model_path: str = "models/best_model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self) -> bool:
        """Load the trained LightGBM model"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            
            # Get feature names (assuming the model was trained with named features)
            if hasattr(self.model, 'feature_name_'):
                self.feature_names = self.model.feature_name_
            else:
                # Fallback: use standard feature names from your dataset
                self.feature_names = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
            
            logger.info(f"✅ Model loaded successfully. Features: {len(self.feature_names)}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def validate_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Validate and ensure all required features are present"""
        validated_features = {}
        
        for feature in self.feature_names:
            if feature in features:
                validated_features[feature] = features[feature]
            else:
                # If feature missing, use 0 as default (you might want different logic)
                logger.warning(f"Feature {feature} not provided, using default value 0")
                validated_features[feature] = 0.0
        
        return validated_features
    
    def predict_single(self, transaction_data: Dict[str, float]) -> Dict[str, Union[float, str]]:
        """Make fraud prediction for a single transaction"""
        try:
            # Validate and prepare features
            validated_features = self.validate_features(transaction_data)
            
            # Create DataFrame with correct feature order
            features_df = pd.DataFrame([validated_features])[self.feature_names]
            
            # Make prediction
            fraud_probability = self.model.predict_proba(features_df)[0][1]  # Probability of class 1 (fraud)
            prediction = self.model.predict(features_df)[0]
            
            # Get feature importance (if available)
            feature_importance = self.get_feature_importance_for_prediction(features_df)
            
            return {
                "fraud_probability": float(fraud_probability),
                "prediction": int(prediction),
                "prediction_label": "Fraud" if prediction == 1 else "Legitimate",
                "confidence": float(fraud_probability) if prediction == 1 else float(1 - fraud_probability),
                "feature_importance": feature_importance,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "fraud_probability": 0.0,
                "prediction": 0,
                "prediction_label": "Error",
                "confidence": 0.0,
                "feature_importance": {},
                "status": f"error: {str(e)}"
            }
    
    def predict_batch(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for multiple transactions"""
        try:
            # Ensure we have all required features
            for feature in self.feature_names:
                if feature not in transactions_df.columns:
                    transactions_df[feature] = 0.0  # Add missing features with default value
            
            # Reorder columns to match training
            features_df = transactions_df[self.feature_names]
            
            # Make predictions
            probabilities = self.model.predict_proba(features_df)[:, 1]
            predictions = self.model.predict(features_df)
            
            # Add results to dataframe
            results_df = transactions_df.copy()
            results_df['fraud_probability'] = probabilities
            results_df['prediction'] = predictions
            results_df['prediction_label'] = results_df['prediction'].apply(
                lambda x: 'Fraud' if x == 1 else 'Legitimate'
            )
            results_df['confidence'] = results_df.apply(
                lambda row: row['fraud_probability'] if row['prediction'] == 1 else 1 - row['fraud_probability'],
                axis=1
            )
            
            return results_df
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise
    
    def get_feature_importance_for_prediction(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Get feature importance for a specific prediction"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                # Get global feature importance
                importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
                
                # Sort by importance
                sorted_importance = dict(
                    sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]  # Top 10 features
                )
                return sorted_importance
            else:
                return {"message": "Feature importance not available"}
                
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
            return {"message": "Feature importance calculation failed"}
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_type": type(self.model).__name__,
            "features_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "model_loaded": True
        }

# Global predictor instance
predictor = FraudPredictor()