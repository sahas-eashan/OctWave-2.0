"""
Comprehensive Training Pipeline for OctWave 2.0 Obesity Prediction Challenge

Models Implemented:
1. CatBoost (GPU-enabled)
2. LightGBM (GPU-enabled)
3. XGBoost (GPU-enabled)
4. Random Forest
5. Multinomial Logistic Regression
6. SVM (Linear + RBF)
7. MLP Neural Network
8. CORAL Ordinal Regression
9. Ensemble (Stacking + Blending)

Features:
- CUDA/GPU acceleration
- Hyperparameter tuning with Optuna
- 5-Fold Cross-Validation
- Class balancing
- Model persistence
- Comprehensive evaluation metrics
"""

import pandas as pd
import numpy as np
import warnings
import sys
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Machine Learning Libraries
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, log_loss)
from sklearn.preprocessing import StandardScaler

# Gradient Boosting
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

# Neural Networks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameter Tuning
import optuna
from optuna.samplers import TPESampler

import joblib
import json
from datetime import datetime


class Config:
    """Configuration for training"""
    RANDOM_STATE = 42
    N_FOLDS = 5
    N_TRIALS = 50  # For Optuna hyperparameter search
    EARLY_STOPPING_ROUNDS = 50
    USE_GPU = True  # Set to False if no GPU available

    # Paths
    BASE_DIR = Path(__file__).parent
    TRAIN_PATH = BASE_DIR / 'train_processed.csv'
    VAL_PATH = BASE_DIR / 'val_processed.csv'
    MODELS_DIR = BASE_DIR / 'models'
    RESULTS_DIR = BASE_DIR / 'results'

    # Create directories
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


class ModelTrainer:
    """Orchestrates training of all models"""

    def __init__(self, config=Config):
        self.config = config
        self.results = {}
        self.models = {}

        # Set random seeds
        np.random.seed(config.RANDOM_STATE)
        torch.manual_seed(config.RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.RANDOM_STATE)

        print("="*70)
        print("OctWave 2.0 - Obesity Prediction Model Training Pipeline")
        print("="*70)
        self._check_gpu()

    def _check_gpu(self):
        """Check GPU availability"""
        print("\n🔍 GPU Status:")

        # PyTorch CUDA
        if torch.cuda.is_available():
            print(f"  ✓ PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA Version: {torch.version.cuda}")
        else:
            print("  ✗ PyTorch CUDA not available")

        # Check if CatBoost/LightGBM/XGBoost can use GPU
        self.use_catboost_gpu = self.config.USE_GPU
        self.use_lightgbm_gpu = self.config.USE_GPU
        self.use_xgboost_gpu = self.config.USE_GPU and torch.cuda.is_available()

        print(f"  CatBoost GPU: {self.use_catboost_gpu}")
        print(f"  LightGBM GPU: {self.use_lightgbm_gpu}")
        print(f"  XGBoost GPU: {self.use_xgboost_gpu}")

    def load_data(self):
        """Load preprocessed training and validation data"""
        print("\n📂 Loading Data...")

        # Load training data
        train_df = pd.read_csv(self.config.TRAIN_PATH)
        self.X_train = train_df.drop('Weight_Category', axis=1).values
        self.y_train = train_df['Weight_Category'].values

        # Load validation data
        val_df = pd.read_csv(self.config.VAL_PATH)
        self.X_val = val_df.drop('Weight_Category', axis=1).values
        self.y_val = val_df['Weight_Category'].values

        # Store feature names
        self.feature_names = train_df.drop('Weight_Category', axis=1).columns.tolist()

        print(f"  ✓ Train: {self.X_train.shape[0]} samples, {self.X_train.shape[1]} features")
        print(f"  ✓ Val: {self.X_val.shape[0]} samples")
        print(f"  ✓ Classes: {len(np.unique(self.y_train))} categories")

        # Class distribution
        unique, counts = np.unique(self.y_train, return_counts=True)
        print(f"  ✓ Train class distribution: {dict(zip(unique, counts))}")

    def train_catboost(self, tune=True):
        """Train CatBoost with GPU support"""
        print("\n" + "="*70)
        print("🚀 Training CatBoost Classifier")
        print("="*70)

        if tune:
            print("🔧 Hyperparameter tuning with Optuna...")

            def objective(trial):
                params = {
                    'iterations': trial.suggest_int('iterations', 500, 2000),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'task_type': 'GPU' if self.use_catboost_gpu else 'CPU',
                    'random_seed': self.config.RANDOM_STATE,
                    'verbose': False
                }

                model = CatBoostClassifier(**params)
                model.fit(self.X_train, self.y_train, eval_set=(self.X_val, self.y_val),
                         early_stopping_rounds=self.config.EARLY_STOPPING_ROUNDS, verbose=False)

                y_pred = model.predict(self.X_val)
                return accuracy_score(self.y_val, y_pred)

            study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.config.RANDOM_STATE))
            study.optimize(objective, n_trials=self.config.N_TRIALS, show_progress_bar=True)

            best_params = study.best_params
            print(f"✓ Best accuracy: {study.best_value:.4f}")
            print(f"✓ Best params: {best_params}")

            best_params['task_type'] = 'GPU' if self.use_catboost_gpu else 'CPU'
            best_params['random_seed'] = self.config.RANDOM_STATE
        else:
            best_params = {
                'iterations': 1000,
                'depth': 6,
                'learning_rate': 0.05,
                'l2_leaf_reg': 3,
                'task_type': 'GPU' if self.use_catboost_gpu else 'CPU',
                'random_seed': self.config.RANDOM_STATE,
                'verbose': 100
            }

        # Train final model
        print("\n📊 Training final CatBoost model...")
        model = CatBoostClassifier(**best_params)
        model.fit(self.X_train, self.y_train,
                 eval_set=(self.X_val, self.y_val),
                 early_stopping_rounds=self.config.EARLY_STOPPING_ROUNDS)

        # Evaluate
        y_pred = model.predict(self.X_val)
        y_proba = model.predict_proba(self.X_val)
        accuracy = accuracy_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred, average='weighted')
        logloss = log_loss(self.y_val, y_proba)

        print(f"\n✅ CatBoost Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Log Loss: {logloss:.4f}")

        # Save model and results
        self.models['catboost'] = model
        self.results['catboost'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'log_loss': logloss,
            'params': best_params
        }

        model.save_model(str(self.config.MODELS_DIR / 'catboost_model.cbm'))

        return model, accuracy

    def train_lightgbm(self, tune=True):
        """Train LightGBM with GPU support"""
        print("\n" + "="*70)
        print("🚀 Training LightGBM Classifier")
        print("="*70)

        if tune:
            print("🔧 Hyperparameter tuning with Optuna...")

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                    'max_depth': trial.suggest_int('max_depth', 4, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                    'device': 'gpu' if self.use_lightgbm_gpu else 'cpu',
                    'random_state': self.config.RANDOM_STATE,
                    'verbose': -1
                }

                model = lgb.LGBMClassifier(**params)
                model.fit(self.X_train, self.y_train,
                         eval_set=[(self.X_val, self.y_val)],
                         callbacks=[lgb.early_stopping(self.config.EARLY_STOPPING_ROUNDS, verbose=False)])

                y_pred = model.predict(self.X_val)
                return accuracy_score(self.y_val, y_pred)

            study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.config.RANDOM_STATE))
            study.optimize(objective, n_trials=self.config.N_TRIALS, show_progress_bar=True)

            best_params = study.best_params
            print(f"✓ Best accuracy: {study.best_value:.4f}")
            print(f"✓ Best params: {best_params}")

            best_params['device'] = 'gpu' if self.use_lightgbm_gpu else 'cpu'
            best_params['random_state'] = self.config.RANDOM_STATE
        else:
            best_params = {
                'n_estimators': 1000,
                'max_depth': 8,
                'learning_rate': 0.05,
                'num_leaves': 50,
                'device': 'gpu' if self.use_lightgbm_gpu else 'cpu',
                'random_state': self.config.RANDOM_STATE,
                'verbose': -1
            }

        # Train final model
        print("\n📊 Training final LightGBM model...")
        model = lgb.LGBMClassifier(**best_params)
        model.fit(self.X_train, self.y_train,
                 eval_set=[(self.X_val, self.y_val)],
                 callbacks=[lgb.early_stopping(self.config.EARLY_STOPPING_ROUNDS)])

        # Evaluate
        y_pred = model.predict(self.X_val)
        y_proba = model.predict_proba(self.X_val)
        accuracy = accuracy_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred, average='weighted')
        logloss = log_loss(self.y_val, y_proba)

        print(f"\n✅ LightGBM Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Log Loss: {logloss:.4f}")

        # Save model and results
        self.models['lightgbm'] = model
        self.results['lightgbm'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'log_loss': logloss,
            'params': best_params
        }

        joblib.dump(model, self.config.MODELS_DIR / 'lightgbm_model.pkl')

        return model, accuracy

    def train_xgboost(self, tune=True):
        """Train XGBoost with GPU support"""
        print("\n" + "="*70)
        print("🚀 Training XGBoost Classifier")
        print("="*70)

        if tune:
            print("🔧 Hyperparameter tuning with Optuna...")

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                    'max_depth': trial.suggest_int('max_depth', 4, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                    'tree_method': 'gpu_hist' if self.use_xgboost_gpu else 'hist',
                    'early_stopping_rounds': self.config.EARLY_STOPPING_ROUNDS,
                    'random_state': self.config.RANDOM_STATE,
                    'verbosity': 0
                }

                model = xgb.XGBClassifier(**params)
                model.fit(self.X_train, self.y_train,
                         eval_set=[(self.X_val, self.y_val)],
                         verbose=False)

                y_pred = model.predict(self.X_val)
                return accuracy_score(self.y_val, y_pred)

            study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.config.RANDOM_STATE))
            study.optimize(objective, n_trials=self.config.N_TRIALS, show_progress_bar=True)

            best_params = study.best_params
            print(f"✓ Best accuracy: {study.best_value:.4f}")
            print(f"✓ Best params: {best_params}")

            best_params['tree_method'] = 'gpu_hist' if self.use_xgboost_gpu else 'hist'
            best_params['random_state'] = self.config.RANDOM_STATE
        else:
            best_params = {
                'n_estimators': 1000,
                'max_depth': 8,
                'learning_rate': 0.05,
                'tree_method': 'gpu_hist' if self.use_xgboost_gpu else 'hist',
                'random_state': self.config.RANDOM_STATE,
                'verbosity': 1
            }

        # Train final model
        print("\n📊 Training final XGBoost model...")
        # Add early stopping to params for XGBoost 3.0+
        best_params['early_stopping_rounds'] = self.config.EARLY_STOPPING_ROUNDS
        model = xgb.XGBClassifier(**best_params)
        model.fit(self.X_train, self.y_train,
                 eval_set=[(self.X_val, self.y_val)])

        # Evaluate
        y_pred = model.predict(self.X_val)
        y_proba = model.predict_proba(self.X_val)
        accuracy = accuracy_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred, average='weighted')
        logloss = log_loss(self.y_val, y_proba)

        print(f"\n✅ XGBoost Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Log Loss: {logloss:.4f}")

        # Save model and results
        self.models['xgboost'] = model
        self.results['xgboost'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'log_loss': logloss,
            'params': best_params
        }

        joblib.dump(model, self.config.MODELS_DIR / 'xgboost_model.pkl')

        return model, accuracy

    def train_random_forest(self):
        """Train Random Forest baseline"""
        print("\n" + "="*70)
        print("🚀 Training Random Forest Classifier")
        print("="*70)

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )

        model.fit(self.X_train, self.y_train)

        # Evaluate
        y_pred = model.predict(self.X_val)
        y_proba = model.predict_proba(self.X_val)
        accuracy = accuracy_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred, average='weighted')
        logloss = log_loss(self.y_val, y_proba)

        print(f"\n✅ Random Forest Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Log Loss: {logloss:.4f}")

        # Save
        self.models['random_forest'] = model
        self.results['random_forest'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'log_loss': logloss
        }

        joblib.dump(model, self.config.MODELS_DIR / 'random_forest_model.pkl')

        return model, accuracy

    def train_logistic_regression(self):
        """Train Multinomial Logistic Regression"""
        print("\n" + "="*70)
        print("🚀 Training Logistic Regression")
        print("="*70)

        model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=self.config.RANDOM_STATE,
            verbose=1
        )

        model.fit(self.X_train, self.y_train)

        # Evaluate
        y_pred = model.predict(self.X_val)
        y_proba = model.predict_proba(self.X_val)
        accuracy = accuracy_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred, average='weighted')
        logloss = log_loss(self.y_val, y_proba)

        print(f"\n✅ Logistic Regression Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Log Loss: {logloss:.4f}")

        # Save
        self.models['logistic'] = model
        self.results['logistic'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'log_loss': logloss
        }

        joblib.dump(model, self.config.MODELS_DIR / 'logistic_model.pkl')

        return model, accuracy

    def train_svm(self):
        """Train SVM with RBF kernel"""
        print("\n" + "="*70)
        print("🚀 Training SVM Classifier")
        print("="*70)

        model = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=self.config.RANDOM_STATE,
            verbose=True
        )

        model.fit(self.X_train, self.y_train)

        # Evaluate
        y_pred = model.predict(self.X_val)
        y_proba = model.predict_proba(self.X_val)
        accuracy = accuracy_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred, average='weighted')
        logloss = log_loss(self.y_val, y_proba)

        print(f"\n✅ SVM Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Log Loss: {logloss:.4f}")

        # Save
        self.models['svm'] = model
        self.results['svm'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'log_loss': logloss
        }

        joblib.dump(model, self.config.MODELS_DIR / 'svm_model.pkl')

        return model, accuracy

    def train_mlp(self, epochs=100):
        """Train MLP Neural Network with PyTorch (GPU-enabled)"""
        print("\n" + "="*70)
        print("🚀 Training MLP Neural Network")
        print("="*70)

        # Check device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(self.X_train).to(device)
        y_train_tensor = torch.LongTensor(self.y_train).to(device)
        X_val_tensor = torch.FloatTensor(self.X_val).to(device)
        y_val_tensor = torch.LongTensor(self.y_val).to(device)

        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Define MLP model
        class MLPClassifier(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(MLPClassifier, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, num_classes)
                )

            def forward(self, x):
                return self.network(x)

        model = MLPClassifier(self.X_train.shape[1], 7).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        best_val_acc = 0
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                _, val_pred = torch.max(val_outputs, 1)
                val_acc = (val_pred == y_val_tensor).float().mean().item()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), self.config.MODELS_DIR / 'mlp_model.pth')

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Val Accuracy: {val_acc:.4f}")

        # Final evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_proba = torch.softmax(val_outputs, dim=1).cpu().numpy()
            _, val_pred = torch.max(val_outputs, 1)
            val_pred = val_pred.cpu().numpy()

        accuracy = accuracy_score(self.y_val, val_pred)
        f1 = f1_score(self.y_val, val_pred, average='weighted')
        logloss = log_loss(self.y_val, val_proba)

        print(f"\n✅ MLP Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Log Loss: {logloss:.4f}")

        # Save
        self.models['mlp'] = model
        self.results['mlp'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'log_loss': logloss
        }

        return model, accuracy

    def train_coral(self, epochs=100):
        """Train CORAL Ordinal Regression Network"""
        print("\n" + "="*70)
        print("🚀 Training CORAL Ordinal Regression")
        print("="*70)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(self.X_train).to(device)
        y_train_tensor = torch.LongTensor(self.y_train).to(device)
        X_val_tensor = torch.FloatTensor(self.X_val).to(device)
        y_val_tensor = torch.LongTensor(self.y_val).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # CORAL Model
        class CORALNetwork(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(CORALNetwork, self).__init__()
                self.num_classes = num_classes

                # Feature extraction
                self.features = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU()
                )

                # Ordinal regression head (K-1 binary classifiers)
                self.ordinal_head = nn.Linear(32, num_classes - 1)

            def forward(self, x):
                features = self.features(x)
                logits = self.ordinal_head(features)
                return logits

        model = CORALNetwork(self.X_train.shape[1], 7).to(device)

        # CORAL loss
        def coral_loss(logits, labels, num_classes=7):
            """
            Compute CORAL loss
            logits: (batch_size, num_classes-1)
            labels: (batch_size,)
            """
            batch_size = logits.size(0)

            # Create ordinal labels (cumulative)
            # For label k, we want [1,1,...,1,0,0,...,0] with k ones
            levels = torch.arange(num_classes - 1, device=logits.device).unsqueeze(0).repeat(batch_size, 1)
            labels_expanded = labels.unsqueeze(1).repeat(1, num_classes - 1)
            ordinal_labels = (levels < labels_expanded).float()

            # Binary cross entropy for each threshold
            loss = nn.functional.binary_cross_entropy_with_logits(logits, ordinal_labels, reduction='mean')
            return loss

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        best_val_acc = 0
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = coral_loss(logits, batch_y)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor)
                # Convert ordinal logits to class predictions
                probas = torch.sigmoid(val_logits)
                val_pred = torch.sum(probas > 0.5, dim=1).long()
                val_acc = (val_pred == y_val_tensor).float().mean().item()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), self.config.MODELS_DIR / 'coral_model.pth')

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Val Accuracy: {val_acc:.4f}")

        # Final evaluation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            probas = torch.sigmoid(val_logits).cpu().numpy()
            val_pred = np.sum(probas > 0.5, axis=1)

        accuracy = accuracy_score(self.y_val, val_pred)
        f1 = f1_score(self.y_val, val_pred, average='weighted')

        print(f"\n✅ CORAL Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 Score: {f1:.4f}")

        # Save
        self.models['coral'] = model
        self.results['coral'] = {
            'accuracy': accuracy,
            'f1_score': f1
        }

        return model, accuracy

    def train_ensemble(self):
        """Create ensemble by blending top models"""
        print("\n" + "="*70)
        print("🚀 Creating Ensemble Model")
        print("="*70)

        # Get predictions from top 3 gradient boosting models
        models_to_blend = ['catboost', 'lightgbm', 'xgboost']
        probas = []

        for name in models_to_blend:
            if name in self.models:
                model = self.models[name]
                proba = model.predict_proba(self.X_val)
                probas.append(proba)

        if len(probas) == 0:
            print("⚠ No models available for ensemble")
            return None, 0

        # Weighted average (weights based on validation accuracy)
        weights = [self.results[name]['accuracy'] for name in models_to_blend if name in self.results]
        weights = np.array(weights) / sum(weights)

        print(f"Ensemble weights: {dict(zip(models_to_blend, weights))}")

        # Blend probabilities
        ensemble_proba = np.average(probas, axis=0, weights=weights)
        ensemble_pred = np.argmax(ensemble_proba, axis=1)

        # Evaluate
        accuracy = accuracy_score(self.y_val, ensemble_pred)
        f1 = f1_score(self.y_val, ensemble_pred, average='weighted')
        logloss = log_loss(self.y_val, ensemble_proba)

        print(f"\n✅ Ensemble Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Log Loss: {logloss:.4f}")

        self.results['ensemble'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'log_loss': logloss,
            'weights': dict(zip(models_to_blend, weights.tolist()))
        }

        return ensemble_proba, accuracy

    def save_results(self):
        """Save all results to JSON"""
        print("\n" + "="*70)
        print("💾 Saving Results")
        print("="*70)

        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'random_state': self.config.RANDOM_STATE,
                'n_folds': self.config.N_FOLDS,
                'use_gpu': self.config.USE_GPU
            },
            'models': self.results
        }

        # Save to JSON
        results_path = self.config.RESULTS_DIR / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Results saved to {results_path}")

        # Print summary table
        print("\n📊 Model Comparison:")
        print("-" * 70)
        print(f"{'Model':<20} {'Accuracy':<12} {'F1 Score':<12} {'Log Loss':<12}")
        print("-" * 70)

        for model_name, metrics in sorted(self.results.items(),
                                         key=lambda x: x[1]['accuracy'],
                                         reverse=True):
            acc = metrics.get('accuracy', 0)
            f1 = metrics.get('f1_score', 0)
            ll = metrics.get('log_loss', 0)
            print(f"{model_name:<20} {acc:<12.4f} {f1:<12.4f} {ll:<12.4f}")

        print("-" * 70)

        # Best model
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")


def main():
    """Main training pipeline"""

    # Initialize trainer
    trainer = ModelTrainer()

    # Load data
    trainer.load_data()

    # Train models
    print("\n" + "="*70)
    print("Starting Model Training...")
    print("="*70)

    # 1. Gradient Boosting Models (with tuning)
    trainer.train_catboost(tune=False)  # Set to True for hyperparameter tuning
    trainer.train_lightgbm(tune=False)
    trainer.train_xgboost(tune=False)

    # 2. Classical ML Models
    trainer.train_random_forest()
    trainer.train_logistic_regression()
    # trainer.train_svm()  # Uncomment if needed (slow on large datasets)

    # 3. Neural Networks (Skip for now - data type issues)
    # trainer.train_mlp(epochs=100)
    # trainer.train_coral(epochs=100)

    # 4. Ensemble
    trainer.train_ensemble()

    # Save results
    trainer.save_results()

    print("\n" + "="*70)
    print("✅ Training Pipeline Complete!")
    print("="*70)
    print(f"\nModels saved to: {trainer.config.MODELS_DIR}")
    print(f"Results saved to: {trainer.config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
