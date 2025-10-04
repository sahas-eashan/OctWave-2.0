@echo off
echo ============================================
echo OctWave 2.0 - Environment Setup
echo ============================================

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing dependencies...
pip install --upgrade pip
pip install pandas numpy scikit-learn
pip install xgboost lightgbm catboost
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install optuna joblib tqdm

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo To activate the environment, run:
echo   venv\Scripts\activate.bat
echo.
echo To start training, run:
echo   python training\train_models.py
echo.
pause
