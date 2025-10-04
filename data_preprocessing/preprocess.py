"""
Comprehensive Data Preprocessing Pipeline for OctWave 2.0 Obesity Prediction Challenge

This script handles:
1. Data cleaning (duplicates, typos, case standardization)
2. Outlier handling and clipping
3. Missing value imputation
4. Feature engineering (BMI, interactions)
5. Categorical encoding (Binary, Ordinal, One-Hot)
6. Feature scaling
7. Stratified train/validation split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import warnings
import sys
warnings.filterwarnings('ignore')

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class ObesityDataPreprocessor:
    """
    Preprocessor for Obesity Risk Prediction dataset
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()

        # Define ordinal mappings
        self.ordinal_mappings = {
            'Snack_Frequency': ['Never', 'Occasionally', 'Often', 'Always'],
            'Alcohol_Consumption': ['no', 'Sometimes', 'Frequently', 'Always'],
            'Physical_Activity_Level': ['None', 'Low', 'Medium', 'High']
        }

    def load_data(self, filepath):
        """Load CSV data"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        return df

    def remove_duplicates(self, df):
        """Remove duplicate rows (excluding PersonID)"""
        print("\n--- Removing Duplicates ---")
        initial_rows = len(df)

        # Check duplicates on all columns except PersonID
        cols_to_check = [col for col in df.columns if col != 'PersonID']
        duplicates = df.duplicated(subset=cols_to_check, keep='first')

        if duplicates.sum() > 0:
            print(f"Found {duplicates.sum()} duplicate rows")
            df = df[~duplicates].reset_index(drop=True)
            print(f"Removed {initial_rows - len(df)} rows")
        else:
            print("No duplicates found")

        return df

    def fix_categorical_issues(self, df):
        """Fix typos, case inconsistencies, and standardize categorical values"""
        print("\n--- Fixing Categorical Issues ---")

        # 1. Standardize High_Calorie_Food to lowercase
        if 'High_Calorie_Food' in df.columns:
            df['High_Calorie_Food'] = df['High_Calorie_Food'].str.lower()
            print("✓ Standardized High_Calorie_Food to lowercase")

        # 2. Fix "yess" typo in Family_History
        if 'Family_History' in df.columns:
            typo_count = (df['Family_History'] == 'yess').sum()
            if typo_count > 0:
                df['Family_History'] = df['Family_History'].replace('yess', 'yes')
                print(f"✓ Fixed {typo_count} 'yess' typos in Family_History")

        # 3. Standardize "Sport" to "Sports"
        if 'Leisure Time Activity' in df.columns:
            sport_count = (df['Leisure Time Activity'] == 'Sport').sum()
            if sport_count > 0:
                df['Leisure Time Activity'] = df['Leisure Time Activity'].replace('Sport', 'Sports')
                print(f"✓ Standardized {sport_count} 'Sport' to 'Sports'")

        # 4. Standardize Alcohol_Consumption (map "no" to fit ordinal scale)
        if 'Alcohol_Consumption' in df.columns:
            # "no" should be treated as lowest level in ordinal scale
            pass  # Already handled in ordinal mapping

        return df

    def handle_outliers(self, df):
        """Handle outliers and invalid values"""
        print("\n--- Handling Outliers & Invalid Values ---")

        # 1. Fix Age = 222 (replace with median) and cap at 100
        if 'Age_Years' in df.columns:
            age_222_count = (df['Age_Years'] == 222).sum()
            if age_222_count > 0:
                median_age = df[df['Age_Years'] < 100]['Age_Years'].median()
                df.loc[df['Age_Years'] == 222, 'Age_Years'] = median_age
                print(f"✓ Fixed {age_222_count} age=222 values (replaced with median: {median_age:.1f})")

            # Cap age at 100
            capped = (df['Age_Years'] > 100).sum()
            if capped > 0:
                df.loc[df['Age_Years'] > 100, 'Age_Years'] = 100
                print(f"✓ Capped {capped} ages > 100 to 100")

        # 2. Clip negative values to 0
        cols_to_clip = ['Screen_Time_Hours', 'Family_Risk', 'Activity_Level_Score']
        for col in cols_to_clip:
            if col in df.columns:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    df[col] = df[col].clip(lower=0)
                    print(f"✓ Clipped {neg_count} negative values in {col} to 0")

        return df

    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("\n--- Handling Missing Values ---")

        # 1. Drop Physical_Activity_Level if 75%+ missing
        if 'Physical_Activity_Level' in df.columns:
            missing_pct = (df['Physical_Activity_Level'].isna().sum() / len(df)) * 100
            if missing_pct > 50:
                df = df.drop('Physical_Activity_Level', axis=1)
                print(f"✓ Dropped Physical_Activity_Level ({missing_pct:.1f}% missing)")

        # 2. Handle missing Gender - impute with mode or drop rows
        if 'Gender' in df.columns:
            missing_gender = df['Gender'].isna().sum()
            if missing_gender > 0:
                mode_gender = df['Gender'].mode()[0]
                df['Gender'].fillna(mode_gender, inplace=True)
                print(f"✓ Imputed {missing_gender} missing Gender values with mode: {mode_gender}")

        # 3. Handle missing Alcohol_Consumption - impute with mode
        if 'Alcohol_Consumption' in df.columns:
            missing_alcohol = df['Alcohol_Consumption'].isna().sum()
            if missing_alcohol > 0:
                mode_alcohol = df['Alcohol_Consumption'].mode()[0]
                df['Alcohol_Consumption'].fillna(mode_alcohol, inplace=True)
                print(f"✓ Imputed {missing_alcohol} missing Alcohol_Consumption values with mode: {mode_alcohol}")

        # 4. Check for any remaining missing values in numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isna().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"✓ Imputed missing {col} with median: {median_val:.2f}")

        return df

    def feature_engineering(self, df):
        """Create new features"""
        print("\n--- Feature Engineering ---")

        # 1. Create BMI
        if 'Weight_Kg' in df.columns and 'Height_cm' in df.columns:
            df['BMI'] = df['Weight_Kg'] / ((df['Height_cm'] / 100) ** 2)
            print(f"✓ Created BMI feature (mean: {df['BMI'].mean():.2f})")

        # 2. Interaction features
        if 'Activity_Level_Score' in df.columns and 'Screen_Time_Hours' in df.columns:
            # Sedentary index: screen time relative to activity
            df['Sedentary_Index'] = df['Screen_Time_Hours'] / (df['Activity_Level_Score'] + 0.1)
            print("✓ Created Sedentary_Index feature")

        if 'Water_Intake' in df.columns and 'Activity_Level_Score' in df.columns:
            df['Hydration_Activity_Ratio'] = df['Water_Intake'] * df['Activity_Level_Score']
            print("✓ Created Hydration_Activity_Ratio feature")

        if 'Snack_Frequency' in df.columns and 'High_Calorie_Food' in df.columns:
            # This will be created after encoding, so mark for later
            pass

        if 'Family_Risk' in df.columns and 'BMI' in df.columns:
            df['Family_Risk_BMI'] = df['Family_Risk'] * df['BMI']
            print("✓ Created Family_Risk_BMI interaction")

        return df

    def encode_features(self, df, fit=True):
        """Encode categorical features"""
        print("\n--- Encoding Categorical Features ---")

        # Separate target if present
        target_col = 'Weight_Category'
        has_target = target_col in df.columns

        if has_target:
            y = df[target_col].copy()
            X = df.drop(target_col, axis=1)
        else:
            X = df.copy()

        # Drop PersonID if present
        if 'PersonID' in X.columns:
            person_ids = X['PersonID'].copy()
            X = X.drop('PersonID', axis=1)

        # 1. Binary encoding (yes/no, Male/Female)
        binary_mappings = {
            'High_Calorie_Food': {'no': 0, 'yes': 1},
            'Gender': {'Female': 0, 'Male': 1},
            'Family_History': {'no': 0, 'yes': 1},
            'Smoking_Habit': {'no': 0, 'yes': 1}
        }

        for col, mapping in binary_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
                print(f"✓ Binary encoded {col}")

        # 2. Ordinal encoding
        ordinal_cols = {
            'Snack_Frequency': {'Never': 0, 'Occasionally': 1, 'Often': 2, 'Always': 3},
            'Alcohol_Consumption': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        }

        for col, mapping in ordinal_cols.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
                print(f"✓ Ordinal encoded {col}")

        # 3. One-Hot encoding for nominal categories
        nominal_cols = ['Commute_Mode', 'Leisure Time Activity']
        cols_to_encode = [col for col in nominal_cols if col in X.columns]

        if cols_to_encode:
            X = pd.get_dummies(X, columns=cols_to_encode, drop_first=False)
            print(f"✓ One-hot encoded: {', '.join(cols_to_encode)}")

        # 4. Encode target variable
        if has_target:
            if fit:
                y_encoded = self.target_encoder.fit_transform(y)
                print(f"✓ Label encoded target: {list(self.target_encoder.classes_)}")
            else:
                y_encoded = self.target_encoder.transform(y)

            return X, y_encoded
        else:
            return X

    def scale_features(self, X, fit=True):
        """Scale numerical features"""
        print("\n--- Scaling Features ---")

        # Identify numerical columns (exclude binary encoded columns)
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Don't scale binary encoded or one-hot encoded features
        binary_cols = ['High_Calorie_Food', 'Gender', 'Family_History', 'Smoking_Habit',
                       'Snack_Frequency', 'Alcohol_Consumption', 'Family_Risk']
        one_hot_cols = [col for col in X.columns if 'Commute_Mode_' in col or 'Leisure Time Activity_' in col]

        cols_to_scale = [col for col in numerical_cols if col not in binary_cols + one_hot_cols]

        if cols_to_scale:
            if fit:
                X[cols_to_scale] = self.scaler.fit_transform(X[cols_to_scale])
                print(f"✓ Scaled {len(cols_to_scale)} numerical features")
            else:
                X[cols_to_scale] = self.scaler.transform(X[cols_to_scale])
            print(f"   Features scaled: {', '.join(cols_to_scale[:5])}...")

        return X

    def train_val_split(self, X, y, test_size=0.2):
        """Create stratified train/validation split"""
        print("\n--- Creating Train/Validation Split ---")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state
        )

        print(f"✓ Train set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
        print(f"✓ Validation set: {X_val.shape[0]} samples ({test_size*100:.0f}%)")

        # Check class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"✓ Train class distribution: {dict(zip(unique, counts))}")

        return X_train, X_val, y_train, y_val

    def fit_transform(self, df, test_size=0.2):
        """Complete preprocessing pipeline for training data"""
        print("="*60)
        print("PREPROCESSING PIPELINE - TRAINING DATA")
        print("="*60)

        # Step 1: Remove duplicates
        df = self.remove_duplicates(df)

        # Step 2: Fix categorical issues
        df = self.fix_categorical_issues(df)

        # Step 3: Handle outliers
        df = self.handle_outliers(df)

        # Step 4: Handle missing values
        df = self.handle_missing_values(df)

        # Step 5: Feature engineering
        df = self.feature_engineering(df)

        # Step 6: Encode features
        X, y = self.encode_features(df, fit=True)

        # Step 7: Scale features
        X = self.scale_features(X, fit=True)

        # Step 8: Train/val split
        X_train, X_val, y_train, y_val = self.train_val_split(X, y, test_size=test_size)

        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Final feature count: {X_train.shape[1]}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")

        return X_train, X_val, y_train, y_val

    def transform(self, df):
        """Transform test data using fitted parameters"""
        print("="*60)
        print("PREPROCESSING PIPELINE - TEST DATA")
        print("="*60)

        # Step 1: Remove duplicates
        df = self.remove_duplicates(df)

        # Step 2: Fix categorical issues
        df = self.fix_categorical_issues(df)

        # Step 3: Handle outliers
        df = self.handle_outliers(df)

        # Step 4: Handle missing values
        df = self.handle_missing_values(df)

        # Step 5: Feature engineering
        df = self.feature_engineering(df)

        # Step 6: Encode features (no target)
        X = self.encode_features(df, fit=False)

        # Step 7: Scale features
        X = self.scale_features(X, fit=False)

        print("\n" + "="*60)
        print("TEST PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Final feature count: {X.shape[1]}")
        print(f"Test samples: {X.shape[0]}")

        return X

    def save_processed_data(self, X_train, X_val, y_train, y_val, output_dir='./processed_data'):
        """Save processed data to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Convert to DataFrames
        train_df = pd.DataFrame(X_train)
        train_df['Weight_Category'] = y_train

        val_df = pd.DataFrame(X_val)
        val_df['Weight_Category'] = y_val

        # Save
        train_path = os.path.join(output_dir, 'train_processed.csv')
        val_path = os.path.join(output_dir, 'val_processed.csv')

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        print(f"\n✓ Saved processed training data to {train_path}")
        print(f"✓ Saved processed validation data to {val_path}")


def main():
    """Main execution function"""

    # Initialize preprocessor
    preprocessor = ObesityDataPreprocessor(random_state=42)

    # Load training data
    train_path = 'train.csv'  # Located in same directory
    df_train = preprocessor.load_data(train_path)

    # Fit and transform training data with 80-20 split
    X_train, X_val, y_train, y_val = preprocessor.fit_transform(df_train, test_size=0.2)

    # Save processed data
    preprocessor.save_processed_data(X_train, X_val, y_train, y_val)

    print("\n✅ Preprocessing pipeline completed successfully!")
    print("\nNext steps:")
    print("1. Train models using X_train, y_train")
    print("2. Validate using X_val, y_val")
    print("3. Use preprocessor.transform() for test data")


if __name__ == "__main__":
    main()
