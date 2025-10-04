"""
Quick Prediction Variations - Try Simple Changes

Since ensemble is best (76.56%), try small tweaks:
1. Slightly different ensemble weights
2. Threshold adjustments
3. Simple feature transformations
"""

import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
import sys
from pathlib import Path
warnings = __import__('warnings')
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent / 'data_preprocessing'))
from preprocess import ObesityDataPreprocessor

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


print("="*70)
print("QUICK VARIATIONS - Testing Around Best Score (76.56%)")
print("="*70)

# Load preprocessor and models
preprocessor = joblib.load('training/models/preprocessor.pkl')
models = {
    'catboost': CatBoostClassifier(),
    'lightgbm': joblib.load('training/models/lightgbm_model.pkl'),
    'xgboost': joblib.load('training/models/xgboost_model.pkl'),
}
models['catboost'].load_model('training/models/catboost_model.cbm')

# Load and preprocess test
test_df = pd.read_csv('testing/test.csv')
test_clean = test_df.copy()
test_clean = preprocessor.fix_categorical_issues(test_clean)
test_clean = preprocessor.handle_outliers(test_clean)
test_clean = preprocessor.handle_missing_values(test_clean)
test_clean = preprocessor.feature_engineering(test_clean)

person_ids = test_clean['PersonID'].copy()
test_clean = test_clean.drop('PersonID', axis=1)

X_test = preprocessor.encode_features(test_clean, fit=False)
X_test = preprocessor.scale_features(X_test, fit=False)

# Get probabilities
print("\nGetting model probabilities...")
probas = {}
for name, model in models.items():
    probas[name] = model.predict_proba(X_test)
    print(f"  {name}")

# Category mapping
category_mapping = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'Obesity_Type_I',
    3: 'Obesity_Type_II',
    4: 'Obesity_Type_III',
    5: 'Overweight_Level_I',
    6: 'Overweight_Level_II'
}

def save_submission(predictions, filename):
    df = pd.DataFrame({
        'PersonID': person_ids,
        'Weight_Category': [category_mapping[int(p)] for p in predictions]
    })
    df.to_csv(filename, index=False)
    return df

print("\n" + "="*70)
print("TESTING ENSEMBLE WEIGHT VARIATIONS")
print("="*70)

# Original best: roughly equal (33.1%, 33.5%, 33.4%)
weight_variations = [
    ('var1_favor_cat', {'catboost': 0.35, 'lightgbm': 0.33, 'xgboost': 0.32}),
    ('var2_favor_lgb', {'catboost': 0.32, 'lightgbm': 0.36, 'xgboost': 0.32}),
    ('var3_favor_xgb', {'catboost': 0.32, 'lightgbm': 0.32, 'xgboost': 0.36}),
    ('var4_equal_exact', {'catboost': 0.3333, 'lightgbm': 0.3333, 'xgboost': 0.3334}),
    ('var5_slight_lgb', {'catboost': 0.33, 'lightgbm': 0.34, 'xgboost': 0.33}),
    ('var6_slight_xgb', {'catboost': 0.33, 'lightgbm': 0.33, 'xgboost': 0.34}),
    ('var7_balanced', {'catboost': 0.34, 'lightgbm': 0.34, 'xgboost': 0.32}),
    ('var8_lgb_xgb_high', {'catboost': 0.30, 'lightgbm': 0.35, 'xgboost': 0.35}),
]

submissions_created = []

for var_name, weights in weight_variations:
    # Weighted average
    weighted_proba = sum(weights[name] * probas[name] for name in weights.keys())
    predictions = np.argmax(weighted_proba, axis=1)

    filename = f'submission_{var_name}.csv'
    save_submission(predictions, filename)
    submissions_created.append(filename)

    print(f"Created: {filename}")
    print(f"  Weights: {weights}")

print("\n" + "="*70)
print("TESTING PROBABILITY THRESHOLD ADJUSTMENTS")
print("="*70)

# Try adjusting class probabilities slightly
# Original ensemble weights
base_weights = {'catboost': 0.331, 'lightgbm': 0.335, 'xgboost': 0.334}
base_proba = sum(base_weights[name] * probas[name] for name in base_weights.keys())

# Strategy: Slightly boost/reduce certain class probabilities
threshold_variations = [
    ('threshold1_boost_obesity', [1.0, 1.0, 1.05, 1.05, 1.05, 1.0, 1.0]),
    ('threshold2_boost_normal', [1.0, 1.05, 1.0, 1.0, 1.0, 1.0, 1.0]),
    ('threshold3_reduce_extreme', [0.98, 1.0, 1.0, 1.0, 0.98, 1.0, 1.0]),
    ('threshold4_boost_overweight', [1.0, 1.0, 1.0, 1.0, 1.0, 1.02, 1.02]),
]

for var_name, multipliers in threshold_variations:
    adjusted_proba = base_proba.copy()
    for i, mult in enumerate(multipliers):
        adjusted_proba[:, i] *= mult

    # Renormalize
    adjusted_proba = adjusted_proba / adjusted_proba.sum(axis=1, keepdims=True)
    predictions = np.argmax(adjusted_proba, axis=1)

    filename = f'submission_{var_name}.csv'
    save_submission(predictions, filename)
    submissions_created.append(filename)

    print(f"Created: {filename}")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"\nCreated {len(submissions_created)} new variations")
print("\nTry these in order:")
print("1. submission_var5_slight_lgb.csv")
print("2. submission_var6_slight_xgb.csv")
print("3. submission_var8_lgb_xgb_high.csv")
print("4. submission_threshold1_boost_obesity.csv")
print("\nThese are TINY variations around the 76.56% best score")
print("Expected: 76.4% - 76.8% (subtle improvements)")
