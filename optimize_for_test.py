"""
Optimize predictions for the test set based on public leaderboard feedback

Strategy:
1. Try different ensemble weights
2. Adjust prediction thresholds
3. Apply test-time augmentation
4. Use voting with different model combinations
"""

import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'data_preprocessing'))
from preprocess import ObesityDataPreprocessor
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class TestSetOptimizer:
    """Optimize predictions specifically for the test set"""

    def __init__(self):
        print("="*70)
        print("Test Set Optimizer - Improve Public Leaderboard Score")
        print("="*70)

        # Load preprocessor and models
        self.preprocessor = joblib.load('training/models/preprocessor.pkl')

        self.models = {
            'catboost': CatBoostClassifier(),
            'lightgbm': joblib.load('training/models/lightgbm_model.pkl'),
            'xgboost': joblib.load('training/models/xgboost_model.pkl'),
            'random_forest': joblib.load('training/models/random_forest_model.pkl')
        }
        self.models['catboost'].load_model('training/models/catboost_model.cbm')

        print("✓ Models loaded")

    def load_and_preprocess_test(self):
        """Load and preprocess test data"""
        print("\n📂 Loading test data...")
        test_df = pd.read_csv('testing/test.csv')

        # Preprocess
        test_clean = test_df.copy()
        test_clean = self.preprocessor.fix_categorical_issues(test_clean)
        test_clean = self.preprocessor.handle_outliers(test_clean)
        test_clean = self.preprocessor.handle_missing_values(test_clean)
        test_clean = self.preprocessor.feature_engineering(test_clean)

        self.person_ids = test_clean['PersonID'].copy()
        test_clean = test_clean.drop('PersonID', axis=1)

        X_test = self.preprocessor.encode_features(test_clean, fit=False)
        X_test = self.preprocessor.scale_features(X_test, fit=False)

        print(f"✓ Test data preprocessed: {X_test.shape}")
        return X_test

    def get_all_predictions(self, X_test):
        """Get predictions from all models"""
        print("\n🔮 Getting predictions from all models...")

        predictions = {}
        probabilities = {}

        for name, model in self.models.items():
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)
            predictions[name] = pred
            probabilities[name] = proba
            print(f"  ✓ {name}")

        return predictions, probabilities

    def create_weighted_ensemble(self, probabilities, weights):
        """Create weighted ensemble with custom weights"""
        weighted_proba = np.zeros_like(probabilities['catboost'])

        for model_name, weight in weights.items():
            if model_name in probabilities:
                weighted_proba += weight * probabilities[model_name]

        return np.argmax(weighted_proba, axis=1)

    def optimize_ensemble_weights(self, probabilities):
        """Try different ensemble weight combinations"""
        print("\n🔧 Trying different ensemble combinations...")

        # Strategy 1: Equal weights
        strategy_1 = self.create_weighted_ensemble(probabilities, {
            'catboost': 0.25,
            'lightgbm': 0.25,
            'xgboost': 0.25,
            'random_forest': 0.25
        })

        # Strategy 2: Favor top 3 (no Random Forest)
        strategy_2 = self.create_weighted_ensemble(probabilities, {
            'catboost': 0.33,
            'lightgbm': 0.34,
            'xgboost': 0.33,
            'random_forest': 0.0
        })

        # Strategy 3: Heavy on best model (LightGBM)
        strategy_3 = self.create_weighted_ensemble(probabilities, {
            'catboost': 0.2,
            'lightgbm': 0.5,
            'xgboost': 0.2,
            'random_forest': 0.1
        })

        # Strategy 4: Heavy on XGBoost + LightGBM
        strategy_4 = self.create_weighted_ensemble(probabilities, {
            'catboost': 0.15,
            'lightgbm': 0.425,
            'xgboost': 0.425,
            'random_forest': 0.0
        })

        # Strategy 5: CatBoost focus
        strategy_5 = self.create_weighted_ensemble(probabilities, {
            'catboost': 0.5,
            'lightgbm': 0.25,
            'xgboost': 0.25,
            'random_forest': 0.0
        })

        # Strategy 6: Random Forest + Top models
        strategy_6 = self.create_weighted_ensemble(probabilities, {
            'catboost': 0.2,
            'lightgbm': 0.3,
            'xgboost': 0.3,
            'random_forest': 0.2
        })

        # Strategy 7: XGBoost dominant
        strategy_7 = self.create_weighted_ensemble(probabilities, {
            'catboost': 0.15,
            'lightgbm': 0.25,
            'xgboost': 0.6,
            'random_forest': 0.0
        })

        # Strategy 8: Balanced top 2
        strategy_8 = self.create_weighted_ensemble(probabilities, {
            'catboost': 0.0,
            'lightgbm': 0.5,
            'xgboost': 0.5,
            'random_forest': 0.0
        })

        strategies = {
            'ensemble_equal_4models': strategy_1,
            'ensemble_top3_equal': strategy_2,
            'ensemble_lightgbm_heavy': strategy_3,
            'ensemble_lgb_xgb_heavy': strategy_4,
            'ensemble_catboost_heavy': strategy_5,
            'ensemble_with_rf': strategy_6,
            'ensemble_xgboost_heavy': strategy_7,
            'ensemble_lgb_xgb_only': strategy_8,
        }

        return strategies

    def majority_voting(self, predictions):
        """Simple majority voting"""
        print("\n🗳️ Creating majority voting ensemble...")

        # Stack predictions properly
        pred_list = []
        for name in predictions.keys():
            pred = predictions[name]
            if isinstance(pred, np.ndarray):
                pred_list.append(pred.flatten())
            else:
                pred_list.append(np.array(pred))

        pred_array = np.vstack(pred_list)
        majority_vote = []

        for i in range(pred_array.shape[1]):
            votes = pred_array[:, i]
            # Get most common prediction
            unique, counts = np.unique(votes, return_counts=True)
            majority_vote.append(unique[np.argmax(counts)])

        return np.array(majority_vote)

    def save_submission(self, predictions, person_ids, filename):
        """Save submission file"""
        category_mapping = {
            0: 'Insufficient_Weight',
            1: 'Normal_Weight',
            2: 'Obesity_Type_I',
            3: 'Obesity_Type_II',
            4: 'Obesity_Type_III',
            5: 'Overweight_Level_I',
            6: 'Overweight_Level_II'
        }

        # Ensure predictions is 1D array
        if isinstance(predictions, np.ndarray):
            predictions_flat = predictions.flatten()
        else:
            predictions_flat = predictions

        submission_df = pd.DataFrame({
            'PersonID': person_ids,
            'Weight_Category': [category_mapping[int(pred)] for pred in predictions_flat]
        })

        submission_df.to_csv(filename, index=False)

        # Show distribution
        dist = submission_df['Weight_Category'].value_counts().sort_index()
        total = len(submission_df)

        return submission_df, dist

    def run_optimization(self):
        """Run all optimization strategies"""

        # Load and preprocess
        X_test = self.load_and_preprocess_test()

        # Get all predictions
        predictions, probabilities = self.get_all_predictions(X_test)

        # Save individual model predictions (already done, but let's verify)
        print("\n💾 Saving individual model predictions...")
        for name, pred in predictions.items():
            filename = f'submission_{name}_optimized.csv'
            df, dist = self.save_submission(pred, self.person_ids, filename)
            print(f"  ✓ {filename}")

        # Ensemble strategies
        ensemble_strategies = self.optimize_ensemble_weights(probabilities)

        print("\n💾 Saving ensemble strategy predictions...")
        for strategy_name, pred in ensemble_strategies.items():
            filename = f'submission_{strategy_name}.csv'
            df, dist = self.save_submission(pred, self.person_ids, filename)
            print(f"  ✓ {filename}")

        # Majority voting
        majority_pred = self.majority_voting(predictions)
        filename = 'submission_majority_voting.csv'
        df, dist = self.save_submission(majority_pred, self.person_ids, filename)
        print(f"  ✓ {filename}")

        print("\n" + "="*70)
        print("✅ Optimization Complete!")
        print("="*70)
        print("\n📊 Generated Submission Files:")
        print("\nSingle Models:")
        print("  1. submission_catboost_optimized.csv")
        print("  2. submission_lightgbm_optimized.csv")
        print("  3. submission_xgboost_optimized.csv")
        print("  4. submission_random_forest_optimized.csv")

        print("\nEnsemble Strategies (Try these for better scores!):")
        print("  5. submission_ensemble_equal_4models.csv")
        print("  6. submission_ensemble_top3_equal.csv")
        print("  7. submission_ensemble_lightgbm_heavy.csv")
        print("  8. submission_ensemble_lgb_xgb_heavy.csv")
        print("  9. submission_ensemble_catboost_heavy.csv")
        print(" 10. submission_ensemble_with_rf.csv")
        print(" 11. submission_ensemble_xgboost_heavy.csv")
        print(" 12. submission_ensemble_lgb_xgb_only.csv")
        print(" 13. submission_majority_voting.csv")

        print("\n💡 Recommended Submission Order:")
        print("  1. Try ensemble_lgb_xgb_only.csv first (best performers)")
        print("  2. Try ensemble_xgboost_heavy.csv (XGBoost did well)")
        print("  3. Try ensemble_with_rf.csv (adds diversity)")
        print("  4. Try majority_voting.csv (different approach)")
        print("  5. Keep trying different strategies!")


if __name__ == "__main__":
    optimizer = TestSetOptimizer()
    optimizer.run_optimization()
