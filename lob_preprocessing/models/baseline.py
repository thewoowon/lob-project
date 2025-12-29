"""
Baseline Models for LOB Mid-Price Prediction

Implements:
1. Logistic Regression
2. XGBoost
3. CatBoost
4. LightGBM
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from typing import Dict, Any, Optional
import time


class BaselineModel:
    """Base class for baseline models"""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.params = kwargs
        self.model = None
        self.training_time = 0
        self.inference_time = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the model"""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        raise NotImplementedError

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importances (if available)"""
        return None


class LogisticModel(BaselineModel):
    """Logistic Regression"""

    def __init__(self, **kwargs):
        default_params = {
            'C': 1.0,
            'max_iter': 1000,
            'solver': 'lbfgs',
            'multi_class': 'multinomial',
            'random_state': 42
        }
        default_params.update(kwargs)
        super().__init__('logistic', **default_params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train logistic regression"""
        start_time = time.time()

        self.model = LogisticRegression(**self.params)
        self.model.fit(X_train, y_train)

        self.training_time = time.time() - start_time
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        start_time = time.time()
        predictions = self.model.predict(X)
        self.inference_time = (time.time() - start_time) / len(X)  # Per sample
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        return self.model.predict_proba(X)


class XGBoostModel(BaselineModel):
    """XGBoost Classifier"""

    def __init__(self, **kwargs):
        default_params = {
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multi:softprob',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        default_params.update(kwargs)
        super().__init__('xgboost', **default_params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train XGBoost"""
        start_time = time.time()

        # Determine number of classes
        num_class = len(np.unique(y_train))
        self.params['num_class'] = num_class

        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X_train, y_train)

        self.training_time = time.time() - start_time
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        start_time = time.time()
        predictions = self.model.predict(X)
        self.inference_time = (time.time() - start_time) / len(X)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances"""
        return self.model.feature_importances_


class CatBoostModel(BaselineModel):
    """CatBoost Classifier"""

    def __init__(self, **kwargs):
        default_params = {
            'depth': 6,
            'learning_rate': 0.1,
            'iterations': 100,
            'verbose': False,
            'random_state': 42,
            'thread_count': -1,
            'loss_function': 'MultiClass'
        }
        default_params.update(kwargs)
        super().__init__('catboost', **default_params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train CatBoost"""
        start_time = time.time()

        self.model = cb.CatBoostClassifier(**self.params)
        self.model.fit(X_train, y_train)

        self.training_time = time.time() - start_time
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        start_time = time.time()
        predictions = self.model.predict(X).flatten()
        self.inference_time = (time.time() - start_time) / len(X)
        return predictions.astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances"""
        return self.model.feature_importances_


class LightGBMModel(BaselineModel):
    """LightGBM Classifier"""

    def __init__(self, **kwargs):
        default_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'objective': 'multiclass'
        }
        default_params.update(kwargs)
        super().__init__('lightgbm', **default_params)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train LightGBM"""
        start_time = time.time()

        # Determine number of classes
        num_class = len(np.unique(y_train))
        self.params['num_class'] = num_class

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X_train, y_train)

        self.training_time = time.time() - start_time
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        start_time = time.time()
        predictions = self.model.predict(X)
        self.inference_time = (time.time() - start_time) / len(X)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances"""
        return self.model.feature_importances_


def get_baseline_model(model_name: str, **kwargs) -> BaselineModel:
    """
    Factory function to get baseline model

    Args:
        model_name: One of ['logistic', 'xgboost', 'catboost', 'lightgbm', 'random_forest']
        **kwargs: Model-specific parameters

    Returns:
        Initialized model
    """
    models = {
        'logistic': LogisticModel,
        'xgboost': XGBoostModel,
        'catboost': CatBoostModel,
        'lightgbm': LightGBMModel,
    }

    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name.lower()](**kwargs)


def main():
    """Example usage"""
    print("=== Baseline Models Module ===\n")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 3, n_samples)  # Ternary classification

    X_test = np.random.randn(200, n_features)
    y_test = np.random.randint(0, 3, 200)

    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Classes: {np.unique(y_train)}\n")

    # Test all models
    models = ['logistic', 'xgboost', 'catboost', 'lightgbm']

    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Testing {model_name.upper()}")
        print('='*50)

        try:
            # Initialize model
            model = get_baseline_model(model_name)

            # Train
            print("Training...")
            model.fit(X_train, y_train)
            print(f"  Training time: {model.training_time:.3f}s")

            # Predict
            print("Predicting...")
            y_pred = model.predict(X_test)
            print(f"  Inference time: {model.inference_time*1000:.3f}ms per sample")

            # Accuracy
            accuracy = np.mean(y_pred == y_test)
            print(f"  Accuracy: {accuracy:.4f}")

            # Feature importance
            if hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance()
                if importance is not None:
                    print(f"  Top 5 features: {np.argsort(importance)[-5:][::-1]}")

            print(f"✓ {model_name} test passed!")

        except Exception as e:
            print(f"✗ {model_name} test failed: {e}")

    print("\n\n✓ All baseline models tested!")


if __name__ == "__main__":
    main()
