"""
Prediction Script for OctWave 2.0 Obesity Risk Prediction Challenge

This script:
1. Loads test data
2. Applies preprocessing pipeline
3. Makes predictions using trained models
4. Creates submission file
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append(str(Path(__file__).parent / 'data_preprocessing'))
sys.path.append(str(Path(__file__).parent / 'training'))

from preprocess import ObesityDataPreprocessor
import joblib
from catboost import CatBoostClassifier

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class ObesityPredictor:
    """Prediction pipeline for test data"""

    def __init__(self):
        self.preprocessor = None
        self.models = {}
        self.target_encoder = None

        print("="*70)
        print("OctWave 2.0 - Obesity Prediction Pipeline")
        print("="*70)

    def load_models(self, models_dir='training/models'):
        """Load trained models and preprocessor"""
        print(f"\n📦 Loading trained models from {models_dir}...")

        models_path = Path(models_dir)

        # Load preprocessor first
        preprocessor_path = models_path / 'preprocessor.pkl'
        if preprocessor_path.exists():
            self.preprocessor = joblib.load(preprocessor_path)
            print("  ✓ Loaded preprocessor")
        else:
            raise FileNotFoundError("Preprocessor not found! Run save_preprocessor.py first")

        # Load CatBoost
        catboost_path = models_path / 'catboost_model.cbm'
        if catboost_path.exists():
            self.models['catboost'] = CatBoostClassifier()
            self.models['catboost'].load_model(str(catboost_path))
            print("  ✓ Loaded CatBoost")

        # Load LightGBM
        lgb_path = models_path / 'lightgbm_model.pkl'
        if lgb_path.exists():
            self.models['lightgbm'] = joblib.load(lgb_path)
            print("  ✓ Loaded LightGBM")

        # Load XGBoost
        xgb_path = models_path / 'xgboost_model.pkl'
        if xgb_path.exists():
            self.models['xgboost'] = joblib.load(xgb_path)
            print("  ✓ Loaded XGBoost")

        # Load Random Forest
        rf_path = models_path / 'random_forest_model.pkl'
        if rf_path.exists():
            self.models['random_forest'] = joblib.load(rf_path)
            print("  ✓ Loaded Random Forest")

        if not self.models:
            raise ValueError("No models found! Please train models first.")

        print(f"  Total models loaded: {len(self.models)}")

    def load_test_data(self, test_path='testing/test.csv'):
        """Load test data"""
        print(f"\n📂 Loading test data from {test_path}...")

        self.test_df = pd.read_csv(test_path)
        self.person_ids = self.test_df['PersonID'].copy()

        print(f"  ✓ Loaded {len(self.test_df)} test samples")
        print(f"  ✓ Features: {self.test_df.shape[1]} columns")

    def preprocess_test_data(self):
        """Preprocess test data using the fitted preprocessor"""
        print("\n🔧 Preprocessing test data...")

        # Step 1-5: Data cleaning (DON'T remove duplicates for test - Kaggle expects all rows)
        test_clean = self.test_df.copy()
        # test_clean = self.preprocessor.remove_duplicates(test_clean)  # Skip for test data!
        test_clean = self.preprocessor.fix_categorical_issues(test_clean)
        test_clean = self.preprocessor.handle_outliers(test_clean)
        test_clean = self.preprocessor.handle_missing_values(test_clean)
        test_clean = self.preprocessor.feature_engineering(test_clean)

        # Save PersonID before encoding
        person_ids = test_clean['PersonID'].copy() if 'PersonID' in test_clean.columns else self.person_ids

        # Drop PersonID
        if 'PersonID' in test_clean.columns:
            test_clean = test_clean.drop('PersonID', axis=1)

        # Encode (no target for test data, use fitted encoders)
        print("  Encoding features...")
        X_test = self.preprocessor.encode_features(test_clean, fit=False)

        # Scale (use fitted scaler)
        print("  Scaling features...")
        X_test = self.preprocessor.scale_features(X_test, fit=False)

        print(f"  ✓ Test data preprocessed: {X_test.shape[0]} samples × {X_test.shape[1]} features")

        return X_test, person_ids

    def predict_single_model(self, X_test, model_name='lightgbm'):
        """Make predictions using a single model"""
        print(f"\n🔮 Making predictions with {model_name}...")

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded!")

        model = self.models[model_name]
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        print(f"  ✓ Predictions complete")

        return predictions, probabilities

    def predict_ensemble(self, X_test, weights=None):
        """Make predictions using ensemble of all models"""
        print("\n🔮 Making ensemble predictions...")

        # Default weights (equal)
        if weights is None:
            weights = {name: 1.0/len(self.models) for name in self.models.keys()}

        print(f"  Ensemble weights: {weights}")

        # Collect probabilities from all models
        all_probas = []
        model_weights = []

        for name, model in self.models.items():
            if name in weights:
                proba = model.predict_proba(X_test)
                all_probas.append(proba)
                model_weights.append(weights[name])

        # Weighted average
        ensemble_proba = np.average(all_probas, axis=0, weights=model_weights)
        ensemble_pred = np.argmax(ensemble_proba, axis=1)

        print(f"  ✓ Ensemble predictions complete")

        return ensemble_pred, ensemble_proba

    def create_submission(self, predictions, person_ids, output_path='submission.csv'):
        """Create submission file"""
        print(f"\n💾 Creating submission file...")

        # Map predictions back to category names
        category_mapping = {
            0: 'Insufficient_Weight',
            1: 'Normal_Weight',
            2: 'Obesity_Type_I',
            3: 'Obesity_Type_II',
            4: 'Obesity_Type_III',
            5: 'Overweight_Level_I',
            6: 'Overweight_Level_II'
        }

        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'PersonID': person_ids,
            'Weight_Category': [category_mapping[pred] for pred in predictions]
        })

        # Save
        submission_df.to_csv(output_path, index=False)

        print(f"  ✓ Submission saved to {output_path}")
        print(f"  ✓ Total predictions: {len(submission_df)}")

        # Show distribution
        print("\n📊 Prediction distribution:")
        dist = submission_df['Weight_Category'].value_counts().sort_index()
        for category, count in dist.items():
            print(f"    {category}: {count}")

        return submission_df


def main():
    """Main prediction pipeline"""

    # Initialize predictor
    predictor = ObesityPredictor()

    # Load trained models
    predictor.load_models()

    # Load test data
    predictor.load_test_data()

    # Preprocess test data
    X_test, person_ids = predictor.preprocess_test_data()

    # Option 1: Single best model (LightGBM)
    print("\n" + "="*70)
    print("Method 1: Single Model Prediction (LightGBM - Best)")
    print("="*70)
    predictions_lgb, probas_lgb = predictor.predict_single_model(X_test, 'lightgbm')
    predictor.create_submission(predictions_lgb, person_ids, 'submission_lightgbm.csv')

    # Option 2: Ensemble prediction
    print("\n" + "="*70)
    print("Method 2: Ensemble Prediction")
    print("="*70)

    # Use weights from training results
    ensemble_weights = {
        'catboost': 0.331,
        'lightgbm': 0.335,
        'xgboost': 0.334
    }

    predictions_ensemble, probas_ensemble = predictor.predict_ensemble(X_test, weights=ensemble_weights)
    predictor.create_submission(predictions_ensemble, person_ids, 'submission_ensemble.csv')

    # Option 3: XGBoost (2nd best)
    print("\n" + "="*70)
    print("Method 3: Single Model Prediction (XGBoost)")
    print("="*70)
    predictions_xgb, probas_xgb = predictor.predict_single_model(X_test, 'xgboost')
    predictor.create_submission(predictions_xgb, person_ids, 'submission_xgboost.csv')

    print("\n" + "="*70)
    print("✅ Prediction Pipeline Complete!")
    print("="*70)
    print("\n📁 Submission files created:")
    print("  1. submission_lightgbm.csv  (Recommended - 79.65% val accuracy)")
    print("  2. submission_ensemble.csv  (Best log loss)")
    print("  3. submission_xgboost.csv   (79.40% val accuracy)")
    print("\n💡 Tip: Try submitting the LightGBM predictions first!")


if __name__ == "__main__":
    main()
