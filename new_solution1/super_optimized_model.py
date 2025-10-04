"""
SUPER OPTIMIZED MODEL - Single Best Approach
Based on proven competition strategies and your leaderboard feedback

Strategy:
1. Minimal preprocessing (only what's necessary)
2. Single powerful model with optimal hyperparameters
3. Focus on HistGradientBoosting (best for mixed data)
4. Add only proven feature engineering (BMI)
5. Tune specifically for this dataset
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


print("="*80)
print("SUPER OPTIMIZED MODEL - Single Best Solution")
print("="*80)

# Paths
TRAIN_PATH = "../data_preprocessing/train.csv"
TEST_PATH = "../testing/test.csv"

print("\n1. LOADING DATA")
print("-"*80)
df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

print(f"Train shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")

# Check class distribution
print("\nClass distribution:")
print(df_train['Weight_Category'].value_counts().sort_index())

print("\n2. FEATURE ENGINEERING")
print("-"*80)

def add_features(df):
    """Add minimal but powerful features"""
    df = df.copy()

    # BMI - proven obesity indicator
    df['BMI'] = df['Weight_Kg'] / ((df['Height_cm'] / 100.0) ** 2)

    # BMI categories (clinical standard)
    df['BMI_category'] = pd.cut(df['BMI'],
                                 bins=[0, 18.5, 25, 30, 35, 40, 100],
                                 labels=[0, 1, 2, 3, 4, 5])

    # Weight-to-height ratio
    df['Weight_Height_Ratio'] = df['Weight_Kg'] / df['Height_cm']

    # Activity index
    df['Total_Activity'] = df['Activity_Level_Score'] * (3 - df['Screen_Time_Hours'])

    return df

df_train = add_features(df_train)
df_test = add_features(df_test)

print("Added features: BMI, BMI_category, Weight_Height_Ratio, Total_Activity")

# Define feature groups
numeric_cols = [
    'Age_Years', 'Weight_Kg', 'Height_cm',
    'Vegetable_Intake', 'Meal_Frequency', 'Water_Intake',
    'Screen_Time_Hours', 'Family_Risk', 'Activity_Level_Score',
    'BMI', 'BMI_category', 'Weight_Height_Ratio', 'Total_Activity'
]

cat_cols = [
    'High_Calorie_Food', 'Gender', 'Family_History',
    'Snack_Frequency', 'Smoking_Habit', 'Alcohol_Consumption',
    'Commute_Mode', 'Physical_Activity_Level', 'Leisure Time Activity'
]

# Prepare data
X_train = df_train[numeric_cols + cat_cols]
y_train = df_train['Weight_Category']
X_test = df_test[numeric_cols + cat_cols]
test_ids = df_test['PersonID']

print(f"\nFeatures: {X_train.shape[1]} total ({len(numeric_cols)} numeric, {len(cat_cols)} categorical)")

print("\n3. PREPROCESSING PIPELINE")
print("-"*80)

# Minimal, effective preprocessing
numeric_tf = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_tf = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_tf, numeric_cols),
    ('cat', cat_tf, cat_cols)
], remainder='drop')

print("Preprocessing: median imputation + standard scaling for numeric")
print("               mode imputation + one-hot for categorical")

print("\n4. MODEL SELECTION & TUNING")
print("-"*80)

# HistGradientBoosting - best for tabular data with mixed types
# Pre-optimized based on validation experiments

best_model = HistGradientBoostingClassifier(
    max_iter=500,
    max_depth=10,
    learning_rate=0.05,
    min_samples_leaf=20,
    l2_regularization=0.1,
    max_bins=255,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=42,
    verbose=0
)

# Create full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])

print("Model: HistGradientBoostingClassifier")
print("Parameters:")
print(f"  max_iter: 500")
print(f"  max_depth: 10")
print(f"  learning_rate: 0.05")
print(f"  min_samples_leaf: 20")
print(f"  l2_regularization: 0.1")
print(f"  early_stopping: True")

print("\n5. CROSS-VALIDATION")
print("-"*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                            scoring='accuracy', n_jobs=-1, verbose=0)

print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"Individual folds: {[f'{s:.4f}' for s in cv_scores]}")

print("\n6. TRAINING ON FULL DATASET")
print("-"*80)

pipeline.fit(X_train, y_train)

# Training accuracy
train_pred = pipeline.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)

print(f"Training accuracy: {train_acc:.4f}")

print("\n7. MAKING PREDICTIONS")
print("-"*80)

test_pred = pipeline.predict(X_test)

print(f"Predictions generated for {len(test_pred)} test samples")

# Prediction distribution
pred_dist = pd.Series(test_pred).value_counts().sort_index()
print("\nPrediction distribution:")
for cat, count in pred_dist.items():
    pct = count / len(test_pred) * 100
    print(f"  {cat:25s}: {count:4d} ({pct:5.2f}%)")

print("\n8. CREATING SUBMISSION")
print("-"*80)

submission = pd.DataFrame({
    'PersonID': test_ids,
    'Weight_Category': test_pred
})

submission.to_csv('submission_super_optimized.csv', index=False)

print(f"Saved: submission_super_optimized.csv")
print(f"Shape: {submission.shape}")
print(f"Format check: {list(submission.columns)}")

# Verify
print("\nFirst 10 predictions:")
print(submission.head(10))

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"\nExpected performance based on CV: {cv_scores.mean():.2%}")
print("\nSubmission file ready: submission_super_optimized.csv")
print("\nThis is a SINGLE optimized model - no ensemble complexity!")

# Save model for future use
import joblib
joblib.dump(pipeline, 'super_optimized_pipeline.pkl')
print("\nModel saved: super_optimized_pipeline.pkl")
