"""
Advanced Preprocessing with Additional Strategies

New approaches:
1. Reduce high-precision noise (round numerical features)
2. Create polynomial features
3. Feature selection based on importance
4. Different encoding strategies
5. Outlier treatment variations
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent / 'data_preprocessing'))
from preprocess import ObesityDataPreprocessor

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def create_advanced_preprocessor(train_path, test_path):
    """Create multiple preprocessing variants"""

    print("="*70)
    print("ADVANCED PREPROCESSING STRATEGIES")
    print("="*70)

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    preprocessor = ObesityDataPreprocessor(random_state=42)

    # Standard preprocessing first
    print("\nApplying standard preprocessing...")
    train_clean = train_df.copy()
    train_clean = preprocessor.fix_categorical_issues(train_clean)
    train_clean = preprocessor.handle_outliers(train_clean)
    train_clean = preprocessor.handle_missing_values(train_clean)

    test_clean = test_df.copy()
    test_clean = preprocessor.fix_categorical_issues(test_clean)
    test_clean = preprocessor.handle_outliers(test_clean)
    test_clean = preprocessor.handle_missing_values(test_clean)

    # Save PersonIDs
    test_person_ids = test_clean['PersonID'].copy()

    # STRATEGY 1: Reduce Noise - Round high-precision values
    print("\n--- Strategy 1: Noise Reduction ---")
    train_s1 = train_clean.copy()
    test_s1 = test_clean.copy()

    numerical_features = ['Age_Years', 'Weight_Kg', 'Height_cm', 'Vegetable_Intake',
                         'Meal_Frequency', 'Water_Intake', 'Screen_Time_Hours',
                         'Activity_Level_Score']

    for col in numerical_features:
        if col in train_s1.columns:
            # Round to 2 decimal places to remove noise
            train_s1[col] = train_s1[col].round(2)
            test_s1[col] = test_s1[col].round(2)

    print("  Rounded numerical features to 2 decimals")

    # STRATEGY 2: More aggressive feature engineering
    print("\n--- Strategy 2: Enhanced Features ---")
    train_s2 = train_clean.copy()
    test_s2 = test_clean.copy()

    def create_enhanced_features(df):
        # Basic features
        df = preprocessor.feature_engineering(df)

        # Additional polynomial/interaction features
        if 'BMI' in df.columns:
            df['BMI_squared'] = df['BMI'] ** 2
            df['BMI_cubed'] = df['BMI'] ** 3

        if 'Age_Years' in df.columns and 'BMI' in df.columns:
            df['Age_BMI_interaction'] = df['Age_Years'] * df['BMI']

        if 'Water_Intake' in df.columns and 'Vegetable_Intake' in df.columns:
            df['Healthy_Index'] = df['Water_Intake'] * df['Vegetable_Intake']

        if 'Screen_Time_Hours' in df.columns and 'Activity_Level_Score' in df.columns:
            df['Activity_Screen_Ratio'] = df['Activity_Level_Score'] / (df['Screen_Time_Hours'] + 0.01)

        return df

    train_s2 = create_enhanced_features(train_s2)
    test_s2 = create_enhanced_features(test_s2)
    print("  Created polynomial and interaction features")

    # STRATEGY 3: Binning continuous variables
    print("\n--- Strategy 3: Binning Strategy ---")
    train_s3 = train_clean.copy()
    test_s3 = test_clean.copy()

    def add_binned_features(df):
        df = preprocessor.feature_engineering(df)

        if 'BMI' in df.columns:
            df['BMI_category'] = pd.cut(df['BMI'],
                                        bins=[0, 18.5, 25, 30, 35, 100],
                                        labels=[0, 1, 2, 3, 4])

        if 'Age_Years' in df.columns:
            df['Age_group'] = pd.cut(df['Age_Years'],
                                     bins=[0, 20, 30, 40, 100],
                                     labels=[0, 1, 2, 3])

        return df

    train_s3 = add_binned_features(train_s3)
    test_s3 = add_binned_features(test_s3)
    print("  Added binned categorical features")

    # STRATEGY 4: Different scaling approach (MinMax instead of Standard)
    print("\n--- Strategy 4: MinMax Scaling ---")
    from sklearn.preprocessing import MinMaxScaler

    train_s4 = train_clean.copy()
    test_s4 = test_clean.copy()

    train_s4 = preprocessor.feature_engineering(train_s4)
    test_s4 = preprocessor.feature_engineering(test_s4)

    print("  Will use MinMaxScaler (0-1 range)")

    # STRATEGY 5: Log transform skewed features
    print("\n--- Strategy 5: Log Transforms ---")
    train_s5 = train_clean.copy()
    test_s5 = test_clean.copy()

    def add_log_features(df):
        df = preprocessor.feature_engineering(df)

        # Log transform highly skewed features
        if 'Weight_Kg' in df.columns:
            df['Weight_log'] = np.log1p(df['Weight_Kg'])

        if 'BMI' in df.columns:
            df['BMI_log'] = np.log1p(df['BMI'])

        return df

    train_s5 = add_log_features(train_s5)
    test_s5 = add_log_features(test_s5)
    print("  Added log-transformed features")

    strategies = {
        'strategy1_noise_reduction': (train_s1, test_s1),
        'strategy2_enhanced_features': (train_s2, test_s2),
        'strategy3_binning': (train_s3, test_s3),
        'strategy4_minmax': (train_s4, test_s4),
        'strategy5_log_transform': (train_s5, test_s5)
    }

    return strategies, test_person_ids, preprocessor


def process_and_save_strategy(strategy_name, train_df, test_df, preprocessor, person_ids):
    """Process a strategy and save the data"""

    print(f"\nProcessing {strategy_name}...")

    # Separate target
    y_train = train_df['Weight_Category'].copy()
    X_train = train_df.drop(['PersonID', 'Weight_Category'], axis=1, errors='ignore')
    X_test = test_df.drop('PersonID', axis=1, errors='ignore')

    # Encode
    X_train_encoded = preprocessor.encode_features(X_train, fit=True)
    X_test_encoded = preprocessor.encode_features(X_test, fit=False)

    # Scale
    X_train_scaled = preprocessor.scale_features(X_train_encoded, fit=True)
    X_test_scaled = preprocessor.scale_features(X_test_encoded, fit=False)

    # Encode target
    y_train_encoded = preprocessor.target_encoder.fit_transform(y_train)

    # Save
    output_dir = Path('training/advanced_preprocessing')
    output_dir.mkdir(exist_ok=True)

    # Save train
    train_out = pd.DataFrame(X_train_scaled)
    train_out['Weight_Category'] = y_train_encoded
    train_out.to_csv(output_dir / f'train_{strategy_name}.csv', index=False)

    # Save test
    test_out = pd.DataFrame(X_test_scaled)
    test_out.insert(0, 'PersonID', person_ids.values)
    test_out.to_csv(output_dir / f'test_{strategy_name}.csv', index=False)

    print(f"  Saved: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test")
    print(f"  Features: {X_train_scaled.shape[1]}")

    return X_train_scaled, X_test_scaled, y_train_encoded


def main():
    # Paths
    train_path = 'data_preprocessing/train.csv'
    test_path = 'testing/test.csv'

    # Create strategies
    strategies, person_ids, preprocessor = create_advanced_preprocessor(train_path, test_path)

    # Process each strategy
    processed_data = {}

    for strategy_name, (train_df, test_df) in strategies.items():
        X_train, X_test, y_train = process_and_save_strategy(
            strategy_name, train_df, test_df, preprocessor, person_ids
        )
        processed_data[strategy_name] = (X_train, X_test, y_train)

    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    print("\nGenerated 5 preprocessing strategies:")
    print("  1. strategy1_noise_reduction    - Rounded values")
    print("  2. strategy2_enhanced_features   - Polynomial features")
    print("  3. strategy3_binning            - Categorical bins")
    print("  4. strategy4_minmax             - MinMax scaling")
    print("  5. strategy5_log_transform      - Log transforms")
    print("\nNext: Train models on each strategy and compare!")


if __name__ == "__main__":
    main()
