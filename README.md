# OctWave 2.0 - Obesity Risk Prediction Challenge

Kaggle competition for predicting obesity weight categories using demographic, lifestyle, and behavioral data.

## 📁 Project Structure

```
OctWave-2.0/
├── data_preprocessing/          # Data analysis and preprocessing scripts
│   ├── preprocess.py           # Main preprocessing pipeline
│   ├── analyze_data.py         # Data analysis script
│   └── preprocessing_requirements.md
│
├── training/                    # Training data and models
│   ├── train_processed.csv     # Processed training data (1,588 samples)
│   ├── val_processed.csv       # Validation data (398 samples)
│   ├── train_models.py         # Comprehensive training pipeline
│   ├── requirements.txt        # Python dependencies
│   └── models/                 # Saved models (auto-created)
│
├── venv/                       # Virtual environment
├── setup_environment.bat       # Setup script for Windows
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Setup Environment

**Windows:**
```bash
setup_environment.bat
```

**Manual Setup:**
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r training/requirements.txt
```

### 2. Data Preprocessing (Already Done ✓)

The data has been preprocessed with the following steps:
- ✓ Removed 9 duplicates
- ✓ Fixed typos (yess→yes, Sport→Sports)
- ✓ Handled outliers (Age=222, negative values)
- ✓ Imputed missing values (Gender, Alcohol_Consumption)
- ✓ Created features (BMI, interactions)
- ✓ Encoded categorical variables
- ✓ Stratified train/val split (80/20)

**Processed data:**
- Training: 1,588 samples × 29 features
- Validation: 398 samples × 29 features

### 3. Train Models

```bash
cd training
python train_models.py
```

## 🤖 Models Implemented

### Gradient Boosting (GPU-enabled)
1. **CatBoost** - Best for categorical data
2. **LightGBM** - Fast and accurate
3. **XGBoost** - Strong competition baseline

### Classical ML
4. **Random Forest** - Robust baseline
5. **Logistic Regression** - Linear baseline
6. **SVM** - RBF kernel

### Neural Networks (CUDA-enabled)
7. **MLP** - Multi-layer perceptron
8. **CORAL** - Ordinal regression for ordered classes

### Ensemble
9. **Weighted Blending** - Combines top 3 models

## ⚙️ Configuration

Edit `train_models.py` to configure:

```python
class Config:
    RANDOM_STATE = 42
    N_FOLDS = 5
    N_TRIALS = 50          # Optuna tuning iterations
    USE_GPU = True         # Enable GPU acceleration
```

### GPU Support

- **CatBoost**: `task_type='GPU'`
- **LightGBM**: `device='gpu'`
- **XGBoost**: `tree_method='gpu_hist'`
- **PyTorch**: Automatic CUDA detection

## 📊 Training Pipeline

The training script (`train_models.py`) includes:

1. **Hyperparameter Tuning** (Optuna)
   - 50 trials per model
   - TPE sampler for efficient search
   - Early stopping

2. **Model Training**
   - Gradient boosting models
   - Classical ML baselines
   - Neural networks (MLP + CORAL)
   - Ensemble blending

3. **Evaluation**
   - Accuracy, F1-Score, Log Loss
   - Confusion matrices
   - Feature importance (tree models)

4. **Model Persistence**
   - Saved to `training/models/`
   - Results logged to JSON

## 📈 Expected Performance

| Model | Expected Accuracy |
|-------|------------------|
| Random Forest | 85-88% |
| Logistic Regression | 80-85% |
| CatBoost | 90-93% |
| LightGBM | 90-93% |
| XGBoost | 90-93% |
| MLP | 88-91% |
| CORAL | 89-92% |
| **Ensemble** | **93-95%** |

## 🔧 Preprocessing Details

### Data Cleaning
- Removed duplicates
- Fixed typos and case inconsistencies
- Clipped outliers and negative values

### Feature Engineering
- **BMI** = Weight / (Height/100)²
- **Sedentary_Index** = Screen_Time / Activity_Level
- **Hydration_Activity_Ratio** = Water_Intake × Activity_Level
- **Family_Risk_BMI** = Family_Risk × BMI

### Encoding
- **Binary**: High_Calorie_Food, Gender, Family_History, Smoking_Habit
- **Ordinal**: Snack_Frequency, Alcohol_Consumption
- **One-Hot**: Commute_Mode, Leisure_Time_Activity

## 🎯 Competition Details

- **Start**: Oct 4, 2025 00:00 AM IST
- **End**: Oct 6, 2025 11:59 PM IST
- **Evaluation**: Accuracy on private test set
- **Submissions**: Max 5 per day
- **Leaderboard**: 30% public, 70% private

## 📝 Weight Categories

7-class classification:
1. Insufficient_Weight
2. Normal_Weight
3. Overweight_Level_I
4. Overweight_Level_II
5. Obesity_Type_I
6. Obesity_Type_II
7. Obesity_Type_III

## 🔗 Resources

- **Competition Guidelines**: `DOC-20251002-WA0037..pdf`
- **Preprocessing Report**: `data_preprocessing/preprocessing_requirements.md`
- **Training Script**: `training/train_models.py`

## 💡 Tips

1. **Enable GPU** for 5-10x faster training
2. **Hyperparameter tune** top models for +1-2% accuracy
3. **Ensemble blending** typically gives best results
4. **CORAL ordinal regression** leverages natural class ordering
5. **Monitor validation** to avoid overfitting

## 📦 Dependencies

```bash
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
torch>=2.0.0
optuna>=3.3.0
joblib>=1.3.0
```

## 🏆 Next Steps

1. ✓ Data preprocessing complete
2. ✓ Training pipeline ready
3. ⏳ Run training: `python training/train_models.py`
4. ⏳ Make predictions on test set
5. ⏳ Submit to Kaggle

---

**Good luck with the competition! 🎉**
