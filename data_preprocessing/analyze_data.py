import pandas as pd
import numpy as np
from collections import Counter

# Read the CSV file
df = pd.read_csv(r'c:\Users\Cyborg\Documents\GitHub\OctWave-2.0\training\train.csv')

print("="*80)
print("COMPREHENSIVE DATA ANALYSIS FOR PREPROCESSING")
print("="*80)

# 1. BASIC INFO
print("\n1. DATASET OVERVIEW")
print("-"*80)
print(f"Total Rows: {len(df)}")
print(f"Total Columns: {len(df.columns)}")
print(f"\nColumn Names:\n{list(df.columns)}")

# 2. DATA TYPES
print("\n2. DATA TYPES")
print("-"*80)
print(df.dtypes)

# 3. MISSING VALUES ANALYSIS
print("\n3. MISSING VALUES ANALYSIS")
print("-"*80)
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
print(missing_data.to_string(index=False))

if len(missing_data) == 0:
    print("No missing values found!")

# 4. CHECK FOR EMPTY STRINGS
print("\n4. EMPTY STRING ANALYSIS")
print("-"*80)
for col in df.columns:
    if df[col].dtype == 'object':
        empty_count = (df[col] == '').sum()
        if empty_count > 0:
            print(f"{col}: {empty_count} empty strings")

# 5. CATEGORICAL VARIABLES ANALYSIS
print("\n5. CATEGORICAL VARIABLES (Object Type)")
print("-"*80)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    unique_vals = df[col].unique()
    print(f"\n{col}:")
    print(f"  Unique Values: {len(unique_vals)}")
    print(f"  Values: {sorted([str(v) for v in unique_vals if pd.notna(v)])}")
    value_counts = df[col].value_counts()
    print(f"  Distribution:\n{value_counts}")

# 6. NUMERICAL VARIABLES ANALYSIS
print("\n6. NUMERICAL VARIABLES STATISTICS")
print("-"*80)
print(df.describe())

# 7. CHECK FOR OUTLIERS (using IQR method)
print("\n7. OUTLIER DETECTION (IQR Method)")
print("-"*80)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    if len(outliers) > 0:
        print(f"\n{col}:")
        print(f"  Range: [{df[col].min()}, {df[col].max()}]")
        print(f"  Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
        print(f"  Outlier bounds: [{lower_bound}, {upper_bound}]")
        print(f"  Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
        print(f"  Sample outlier values: {outliers.head(10).tolist()}")

# 8. DATA CONSISTENCY CHECKS
print("\n8. DATA CONSISTENCY CHECKS")
print("-"*80)

# Check for typos in binary/categorical columns
print("\nChecking for inconsistencies in categorical columns:")
for col in categorical_cols:
    unique_vals = [str(v).lower() for v in df[col].dropna().unique()]
    # Check for similar values
    unique_vals_set = set(unique_vals)
    if col in ['High_Calorie_Food', 'Family_History']:
        expected = ['yes', 'no']
        unexpected = [v for v in unique_vals_set if v not in expected]
        if unexpected:
            print(f"\n{col} - Unexpected values: {unexpected}")
            for val in unexpected:
                count = df[col].str.lower().eq(val).sum()
                print(f"  '{val}': {count} occurrences")
                # Show sample rows
                sample_rows = df[df[col].str.lower() == val].head(3)[['PersonID', col]]
                print(f"  Sample rows:\n{sample_rows}")

# 9. CHECK FOR NEGATIVE VALUES WHERE NOT EXPECTED
print("\n9. NEGATIVE VALUES CHECK")
print("-"*80)
for col in numerical_cols:
    if col not in ['Activity_Level_Score', 'Screen_Time_Hours', 'Water_Intake']:  # These might legitimately have negatives
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            print(f"{col}: {negative_count} negative values")
            print(f"  Sample values: {df[df[col] < 0][col].head(10).tolist()}")

# 10. UNUSUAL VALUES
print("\n10. UNUSUAL VALUES DETECTION")
print("-"*80)

# Age check
print("\nAge_Years:")
print(f"  Min: {df['Age_Years'].min()}, Max: {df['Age_Years'].max()}")
if df['Age_Years'].max() > 100:
    extreme_age = df[df['Age_Years'] > 100]
    print(f"  ALERT: {len(extreme_age)} rows with age > 100")
    print(f"  Sample: {extreme_age[['PersonID', 'Age_Years']].head()}")

# Height check
print("\nHeight_cm:")
print(f"  Min: {df['Height_cm'].min()}, Max: {df['Height_cm'].max()}")
if df['Height_cm'].min() < 100 or df['Height_cm'].max() > 250:
    print(f"  ALERT: Unusual height values detected")

# Weight check
print("\nWeight_Kg:")
print(f"  Min: {df['Weight_Kg'].min()}, Max: {df['Weight_Kg'].max()}")

# 11. MIXED DATA TYPES CHECK
print("\n11. MIXED DATA TYPE CHECK")
print("-"*80)
for col in df.columns:
    # Try to identify columns that might have mixed types
    sample_values = df[col].dropna().head(100)
    types = set([type(v).__name__ for v in sample_values])
    if len(types) > 1:
        print(f"{col}: Multiple types detected - {types}")

# 12. FEATURE DISTRIBUTIONS
print("\n12. FEATURE VALUE DISTRIBUTIONS (First few columns)")
print("-"*80)
for col in df.columns[:10]:
    print(f"\n{col}:")
    if df[col].dtype == 'object':
        print(df[col].value_counts(dropna=False))
    else:
        print(f"  Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")
        print(f"  Min: {df[col].min():.2f}, Max: {df[col].max():.2f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
