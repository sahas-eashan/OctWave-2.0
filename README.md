# Comprehensive Data Preprocessing Requirements

**Dataset:** c:\Users\Cyborg\Documents\GitHub\OctWave-2.0\training\train.csv
**Total Records:** 1,995 rows
**Total Features:** 20 columns

---

## 1. MISSING VALUES

### Critical Issues:

1. **Physical_Activity_Level** - **75.09% missing** (1,498 out of 1,995 records)
   - This is a severe data quality issue
   - **Recommendation:** Either drop this column entirely or implement advanced imputation techniques
   - Most values appear to be missing systematically (not random)

2. **Alcohol_Consumption** - **1.85% missing** (37 records)
   - Missing values: NaN
   - **Recommendation:** Impute with mode ("Sometimes") or create a "Unknown" category

3. **Gender** - **1.50% missing** (30 records)
   - **Example rows:** P355 (row 15), P043 (row 43), P069 (row 69), P1959 (row 69), P1372 (row 92), P005 (row 1934)
   - **Recommendation:** Drop these rows or impute based on other features if possible

---

## 2. DATA TYPE INCONSISTENCIES

### Binary/Categorical Fields with Capitalization Issues:

1. **High_Calorie_Food** - Mixed case values:
   - "yes": 1,745 records (87.5%)
   - "no": 225 records (11.3%)
   - "Yes": 16 records (0.8%)
   - "No": 9 records (0.5%)
   - **Action Required:** Standardize to lowercase ("yes"/"no")

2. **Family_History** - Contains typo:
   - "yes": 1,622 records
   - "no": 370 records
   - **"yess": 3 records** (TYPO found in rows: P2143, P965, P1309)
   - **Action Required:** Fix typo "yess" to "yes"

3. **Leisure Time Activity** - Inconsistent naming:
   - "Sports": 412 records
   - "Sport": 4 records (inconsistent)
   - **Action Required:** Standardize "Sport" to "Sports"

---

## 3. CATEGORICAL VARIABLES REQUIRING ENCODING

### Variables that need encoding:

1. **High_Calorie_Food** - Binary (yes/no) - Use Label Encoding or Binary Encoding
2. **Gender** - Binary (Male/Female) - Use Label Encoding
3. **Family_History** - Binary (yes/no) - Use Label Encoding
4. **Smoking_Habit** - Binary (yes/no) - Use Label Encoding
5. **Snack_Frequency** - Ordinal (Never, Occasionally, Often, Always) - Use Ordinal Encoding
6. **Alcohol_Consumption** - Ordinal (no, Sometimes, Frequently, Always) - Use Ordinal Encoding
7. **Commute_Mode** - Nominal (Public_Transportation, Automobile, Walking, Motorbike, Bike) - Use One-Hot Encoding
8. **Weight_Category** - Target Variable (7 classes) - Use Label Encoding
9. **Physical_Activity_Level** - Ordinal (None, Low, Medium, High) - Use Ordinal Encoding (if not dropped)
10. **Leisure Time Activity** - Nominal (Reading, Gaming, Music, Painting, Sports) - Use One-Hot Encoding

### Encoding Recommendations:

**Binary Variables (Label Encoding: 0/1):**
- High_Calorie_Food: no=0, yes=1
- Gender: Female=0, Male=1
- Family_History: no=0, yes=1
- Smoking_Habit: no=0, yes=1

**Ordinal Variables:**
- Snack_Frequency: Never=0, Occasionally=1, Often=2, Always=3
- Alcohol_Consumption: no=0, Sometimes=1, Frequently=2, Always=3
- Physical_Activity_Level: None=0, Low=1, Medium=2, High=3

**Nominal Variables (One-Hot Encoding):**
- Commute_Mode (5 categories)
- Leisure Time Activity (5 categories after standardization)

---

## 4. OUTLIERS DETECTION

### Extreme Outliers:

1. **Age_Years** - CRITICAL ISSUE:
   - Range: [14.0, 222.0]
   - **ALERT: 1 person with age 222 years** (PersonID: P063)
   - **ALERT: 1 person with age 101 years** (PersonID: P287)
   - 160 records (8.02%) flagged as outliers (age > 35.11 years)
   - **Action Required:**
     - Investigate age=222 (likely data entry error)
     - Consider capping age at reasonable maximum (e.g., 100)

2. **Weight_Kg**:
   - Range: [39.37, 173.0]
   - 1 outlier at 173 kg (0.05%)
   - **Action Required:** Acceptable outlier, no action needed

3. **Height_cm**:
   - Range: [145.0, 198.0]
   - 1 outlier at 198 cm (0.05%)
   - **Action Required:** Acceptable outlier, no action needed

4. **Meal_Frequency**:
   - Range: [0.90, 4.06]
   - 592 outliers (29.67%) - values outside [2.38, 3.37]
   - **Action Required:** These appear to be valid variations; no action needed

5. **Family_Risk**:
   - Range: [-0.05, 1.06]
   - 528 outliers (26.47%)
   - **35 negative values detected** (e.g., -0.048683513, -0.021481672)
   - **Action Required:** Investigate negative values; may need clipping to 0

6. **Activity_Level_Score**:
   - Range: [-0.13, 5.0]
   - 24 outliers (1.20%) with scores > 4.31
   - **1 negative value** (-0.128487427)
   - **Action Required:** Review negative value; may need clipping to 0

---

## 5. UNUSUAL/EXTREME VALUES

### Critical Issues:

1. **Age_Years = 222** (PersonID: P063)
   - Clearly an error - humanly impossible
   - **Recommendation:** Remove this record or replace with median age

2. **Age_Years = 101** (PersonID: P287)
   - Unlikely but possible
   - **Recommendation:** Review for validity

3. **Screen_Time_Hours**:
   - Contains negative values (minimum: -0.09)
   - **Action Required:** Clip negative values to 0

4. **Water_Intake**:
   - Contains values near 1.0 and up to 3.05
   - Appears to be measured in liters
   - **Action Required:** No immediate action, but verify units

---

## 6. NUMERICAL FEATURE DISTRIBUTIONS

### Feature Statistics:

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| Age_Years | 24.46 | 8.01 | 14.0 | 222.0 |
| Weight_Kg | 86.91 | 26.47 | 39.37 | 173.0 |
| Height_cm | 170.46 | 9.39 | 145.0 | 198.0 |
| Vegetable_Intake | 2.43 | 0.53 | 1.0 | 3.06 |
| Meal_Frequency | 2.70 | 0.78 | 0.90 | 4.06 |
| Water_Intake | 2.03 | 0.62 | 0.92 | 3.05 |
| Screen_Time_Hours | 0.66 | 0.61 | -0.09 | 2.06 |
| Family_Risk | 0.82 | 0.39 | -0.05 | 1.06 |
| Activity_Level_Score | 1.66 | 1.07 | -0.13 | 5.0 |

**Note:** Most features appear to have reasonable distributions except for Age_Years outliers.

---

## 7. DATA QUALITY ISSUES SUMMARY

### High Priority Issues:

1. **Physical_Activity_Level** - 75% missing - Consider dropping
2. **Age = 222 years** - Data entry error - Must fix
3. **"yess" typo** in Family_History - Must fix
4. **Capitalization inconsistencies** - Must standardize
5. **Negative values** in Family_Risk and Screen_Time_Hours - Must clip to 0
6. **Missing Gender values** (30 records) - Handle appropriately
7. **Missing Alcohol_Consumption** (37 records) - Impute or flag

### Medium Priority Issues:

1. **"Sport" vs "Sports"** inconsistency - Standardize
2. **Age outliers** (>35 years, 160 records) - Review and decide on handling
3. **Meal_Frequency outliers** - Likely valid, monitor

### Low Priority Issues:

1. **PersonID** - Unique identifier, can be dropped for modeling
2. **Height/Weight single outliers** - Acceptable, no action

---

## 8. RECOMMENDED PREPROCESSING PIPELINE

### Step 1: Data Cleaning
```python
1. Standardize High_Calorie_Food to lowercase
2. Fix "yess" to "yes" in Family_History
3. Standardize "Sport" to "Sports" in Leisure Time Activity
4. Handle Age = 222 (remove row or replace with median)
5. Clip negative values in Screen_Time_Hours to 0
6. Clip negative values in Family_Risk to 0
7. Clip negative values in Activity_Level_Score to 0
```

### Step 2: Handle Missing Values
```python
1. Drop Physical_Activity_Level column (75% missing)
2. Impute or drop 30 rows with missing Gender
3. Impute 37 missing Alcohol_Consumption values with mode
```

### Step 3: Feature Engineering
```python
1. Create BMI feature: Weight_Kg / (Height_cm/100)^2
2. Create age groups if needed
3. Consider interaction features
```

### Step 4: Encoding
```python
1. Label encode binary variables (High_Calorie_Food, Gender, Family_History, Smoking_Habit)
2. Ordinal encode (Snack_Frequency, Alcohol_Consumption)
3. One-hot encode (Commute_Mode, Leisure Time Activity)
4. Label encode target (Weight_Category)
```

### Step 5: Scaling/Normalization
```python
1. StandardScaler or MinMaxScaler for numerical features
2. Keep categorical encoded features as-is
```

### Step 6: Train-Test Split
```python
1. Stratified split to maintain class balance
2. 80-20 or 70-30 split recommended
```

---

## 9. SPECIFIC EXAMPLES FROM DATA

### Example of Typo (Family_History = "yess"):
- Row 46: PersonID P2143 - Family_History: "yess"
- Row 1965: PersonID P965 - Family_History: "yess"
- Row 1972: PersonID P1309 - Family_History: "yess"

### Example of Case Inconsistency (High_Calorie_Food):
- Row 5: PersonID P2201 - High_Calorie_Food: "Yes" (should be "yes")
- Row 514: PersonID P991 - High_Calorie_Food: "No" (should be "no")

### Example of Age Outlier:
- Row 1091: PersonID P063 - Age_Years: 222.0 (IMPOSSIBLE - ERROR)
- Row 1138: PersonID P287 - Age_Years: 101.0 (UNLIKELY)

### Example of Missing Values:
- Row 15: PersonID P355 - Gender: NaN
- Row 16: PersonID P019 - Alcohol_Consumption: NaN
- Row 3: PersonID P1021 - Physical_Activity_Level: NaN (75% of dataset)

---

## 10. EXPECTED PREPROCESSING IMPACT

**Original Dataset:** 1,995 rows × 20 columns

**After Cleaning (Estimated):**
- If dropping rows with missing Gender: ~1,965 rows
- If dropping Physical_Activity_Level column: 19 columns
- After one-hot encoding Commute_Mode (5 categories): +4 columns
- After one-hot encoding Leisure Time Activity (5 categories): +4 columns
- Drop PersonID for modeling: -1 column

**Expected Final Shape:** ~1,965 rows × 26 columns (before feature selection)

---

## 11. CONCLUSION

This dataset requires **moderate to significant preprocessing** before modeling:

1. **Critical Issues:** 75% missing data in Physical_Activity_Level, age=222 error, typos
2. **Data Quality:** Generally good for most features
3. **Encoding Required:** 10 categorical variables need encoding
4. **Outliers:** A few extreme values that need attention
5. **Missing Values:** 3 columns with missing data

**Recommended Action:** Implement the 6-step preprocessing pipeline outlined above before training any machine learning models.
