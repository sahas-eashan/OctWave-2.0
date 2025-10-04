"""
Save the fitted preprocessor for use in predictions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'data_preprocessing'))

from preprocess import ObesityDataPreprocessor
import joblib

# Initialize preprocessor
preprocessor = ObesityDataPreprocessor(random_state=42)

# Load and fit on training data
print("Loading training data...")
df_train = preprocessor.load_data('training/train_processed.csv')

# We need to fit the preprocessor on the ORIGINAL training data, not processed
# Let's use a simpler approach - load original and preprocess

df_train_original = preprocessor.load_data('data_preprocessing/train.csv')

print("\nFitting preprocessor...")
# Fit without splitting
df_clean = preprocessor.remove_duplicates(df_train_original)
df_clean = preprocessor.fix_categorical_issues(df_clean)
df_clean = preprocessor.handle_outliers(df_clean)
df_clean = preprocessor.handle_missing_values(df_clean)
df_clean = preprocessor.feature_engineering(df_clean)

# Encode
X, y = preprocessor.encode_features(df_clean, fit=True)

# Scale
X_scaled = preprocessor.scale_features(X, fit=True)

print("\nSaving preprocessor...")
joblib.dump(preprocessor, 'training/models/preprocessor.pkl')
print("✓ Preprocessor saved to training/models/preprocessor.pkl")
