import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv(r'c:\Users\Cyborg\Documents\GitHub\OctWave-2.0\training\train.csv')

print("="*80)
print("ADDITIONAL DATA CHECKS")
print("="*80)

# 1. Check for duplicate rows
print("\n1. DUPLICATE ROWS CHECK")
print("-"*80)
duplicate_rows = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")
if duplicate_rows > 0:
    print("\nDuplicate rows:")
    print(df[df.duplicated(keep=False)])

# 2. Check for duplicate PersonIDs
print("\n2. DUPLICATE PERSON IDs CHECK")
print("-"*80)
duplicate_ids = df['PersonID'].duplicated().sum()
print(f"Number of duplicate PersonIDs: {duplicate_ids}")
if duplicate_ids > 0:
    print("\nDuplicate IDs:")
    print(df[df['PersonID'].duplicated(keep=False)].sort_values('PersonID'))

# 3. Check value ranges for specific features
print("\n3. FEATURE VALUE RANGE VALIDATION")
print("-"*80)

# Check if Vegetable_Intake, Meal_Frequency have reasonable ranges
print("\nVegetable_Intake:")
print(f"  Range: [{df['Vegetable_Intake'].min():.2f}, {df['Vegetable_Intake'].max():.2f}]")
print(f"  Values > 3: {(df['Vegetable_Intake'] > 3).sum()}")
print(f"  Values < 0: {(df['Vegetable_Intake'] < 0).sum()}")

print("\nMeal_Frequency:")
print(f"  Range: [{df['Meal_Frequency'].min():.2f}, {df['Meal_Frequency'].max():.2f}]")
print(f"  Values > 4: {(df['Meal_Frequency'] > 4).sum()}")
print(f"  Values < 1: {(df['Meal_Frequency'] < 1).sum()}")

print("\nWater_Intake:")
print(f"  Range: [{df['Water_Intake'].min():.2f}, {df['Water_Intake'].max():.2f}]")
print(f"  Values > 3: {(df['Water_Intake'] > 3).sum()}")
print(f"  Values < 0: {(df['Water_Intake'] < 0).sum()}")

print("\nScreen_Time_Hours:")
print(f"  Range: [{df['Screen_Time_Hours'].min():.2f}, {df['Screen_Time_Hours'].max():.2f}]")
print(f"  Values < 0: {(df['Screen_Time_Hours'] < 0).sum()}")
if (df['Screen_Time_Hours'] < 0).sum() > 0:
    print(f"  Negative values: {df[df['Screen_Time_Hours'] < 0]['Screen_Time_Hours'].tolist()}")

# 4. Check for whitespace issues in string columns
print("\n4. WHITESPACE ISSUES CHECK")
print("-"*80)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    # Check for leading/trailing whitespace
    has_whitespace = df[col].astype(str).str.strip() != df[col].astype(str)
    whitespace_count = has_whitespace.sum()
    if whitespace_count > 0:
        print(f"{col}: {whitespace_count} values with leading/trailing whitespace")

# 5. Check class balance for target variable
print("\n5. TARGET VARIABLE CLASS BALANCE")
print("-"*80)
print("\nWeight_Category distribution:")
weight_dist = df['Weight_Category'].value_counts()
weight_pct = df['Weight_Category'].value_counts(normalize=True) * 100
for category in weight_dist.index:
    print(f"  {category}: {weight_dist[category]} ({weight_pct[category]:.2f}%)")

# 6. Check for impossible BMI values
print("\n6. BMI CALCULATION AND VALIDATION")
print("-"*80)
df['BMI'] = df['Weight_Kg'] / ((df['Height_cm'] / 100) ** 2)
print(f"\nBMI Statistics:")
print(f"  Mean: {df['BMI'].mean():.2f}")
print(f"  Std: {df['BMI'].std():.2f}")
print(f"  Min: {df['BMI'].min():.2f}")
print(f"  Max: {df['BMI'].max():.2f}")
print(f"\nBMI Categories (Standard):")
print(f"  Underweight (BMI < 18.5): {(df['BMI'] < 18.5).sum()} ({(df['BMI'] < 18.5).sum()/len(df)*100:.2f}%)")
print(f"  Normal (18.5 <= BMI < 25): {((df['BMI'] >= 18.5) & (df['BMI'] < 25)).sum()} ({((df['BMI'] >= 18.5) & (df['BMI'] < 25)).sum()/len(df)*100:.2f}%)")
print(f"  Overweight (25 <= BMI < 30): {((df['BMI'] >= 25) & (df['BMI'] < 30)).sum()} ({((df['BMI'] >= 25) & (df['BMI'] < 30)).sum()/len(df)*100:.2f}%)")
print(f"  Obese (BMI >= 30): {(df['BMI'] >= 30).sum()} ({(df['BMI'] >= 30).sum()/len(df)*100:.2f}%)")

# 7. Check correlation between Weight_Category and BMI
print("\n7. WEIGHT CATEGORY vs BMI ALIGNMENT")
print("-"*80)
print("\nAverage BMI by Weight_Category:")
for category in sorted(df['Weight_Category'].unique()):
    avg_bmi = df[df['Weight_Category'] == category]['BMI'].mean()
    count = len(df[df['Weight_Category'] == category])
    print(f"  {category}: {avg_bmi:.2f} (n={count})")

# 8. Check for zero variance features
print("\n8. ZERO/LOW VARIANCE FEATURES")
print("-"*80)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numerical_cols:
    variance = df[col].var()
    if variance < 0.01:
        print(f"{col}: Very low variance ({variance:.6f})")

# 9. Check decimal precision patterns
print("\n9. DECIMAL PRECISION ANALYSIS")
print("-"*80)
print("\nChecking for integer vs float patterns:")
for col in ['Age_Years', 'Weight_Kg', 'Height_cm', 'Vegetable_Intake', 'Meal_Frequency']:
    is_integer = (df[col] == df[col].astype(int)).sum()
    is_float = len(df) - is_integer
    print(f"{col}:")
    print(f"  Integer values: {is_integer} ({is_integer/len(df)*100:.1f}%)")
    print(f"  Float values: {is_float} ({is_float/len(df)*100:.1f}%)")

# 10. Check for patterns in missing Physical_Activity_Level
print("\n10. MISSING PHYSICAL_ACTIVITY_LEVEL PATTERNS")
print("-"*80)
print(f"\nTotal missing: {df['Physical_Activity_Level'].isnull().sum()} ({df['Physical_Activity_Level'].isnull().sum()/len(df)*100:.2f}%)")
print(f"\nChecking if missing values correlate with other features:")

# Check if missing Physical_Activity_Level correlates with specific values of other categorical features
for col in ['Gender', 'Weight_Category', 'Commute_Mode']:
    print(f"\n{col} distribution when Physical_Activity_Level is missing:")
    missing_mask = df['Physical_Activity_Level'].isnull()
    if missing_mask.sum() > 0:
        dist = df[missing_mask][col].value_counts().head(5)
        print(dist)

print("\n" + "="*80)
print("ADDITIONAL CHECKS COMPLETE")
print("="*80)
