"""
Model Training and Evaluation Module
Handles model training, evaluation, and persistence for loan prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Any, Tuple, List, Optional
import logging
import joblib
from pathlib import Path
import json
from datetime import datetime

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from config_manager import get_config


class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model_type: str = 'loan_outcome_model'):
        """
        Initialize model trainer
        
        Args:
            model_type: Type of model to train ('loan_outcome_model' or 'denial_reason_model')
        """
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type
        self.model = None
        self.model_params = self.config.get_model_params(model_type)
        self.algorithm = self.config.get(f'{model_type}.algorithm', 'random_forest')
        
    def _get_model(self) -> Any:
        """
        Get model instance based on configuration
        
        Returns:
            Initialized model instance
        """
        if self.algorithm == 'random_forest':
            return RandomForestClassifier(**self.model_params)
        
        elif self.algorithm == 'logistic_regression':
            return LogisticRegression(**self.model_params)
        
        elif self.algorithm == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed. Please install it with: pip install xgboost")
            return xgb.XGBClassifier(**self.model_params)
        
        elif self.algorithm == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM is not installed. Please install it with: pip install lightgbm")
            return lgb.LGBMClassifier(**self.model_params)
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training results dictionary
        """
        self.logger.info(f"Training {self.algorithm} model for {self.model_type}")
        
        # Initialize model
        self.model = self._get_model()
        
        # Train model
        if self.algorithm in ['xgboost', 'lightgbm'] and X_val is not None:
            # Use validation set for early stopping if available
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        self.logger.info("Model training completed")
        
        # Perform cross-validation if enabled
        training_results = {}
        training_config = self.config.get_training_config()
        
        if training_config.get('cross_validation', {}).get('enabled', False):
            cv_results = self._perform_cross_validation(X_train, y_train)
            training_results['cross_validation'] = cv_results
        
        return training_results
    
    def _perform_cross_validation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Targets
            
        Returns:
            Cross-validation results
        """
        training_config = self.config.get_training_config()
        cv_config = training_config.get('cross_validation', {})
        
        cv_folds = cv_config.get('cv_folds', 5)
        scoring = cv_config.get('scoring', 'f1_weighted')
        
        self.logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        cv_model = self._get_model()
        scores = cross_val_score(cv_model, X, y, cv=cv_folds, scoring=scoring)
        
        cv_results = {
            'mean_score': float(scores.mean()),
            'std_score': float(scores.std()),
            'scores': scores.tolist()
        }
        
        self.logger.info(f"CV Score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
        
        return cv_results
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation results dictionary
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        self.logger.info("Evaluating model performance")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = None
        
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        results = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Add ROC AUC for binary classification
        if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
            results['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba[:, 1]))
        
        # Log key metrics
        self.logger.info(f"Model Evaluation Results:")
        self.logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        self.logger.info(f"  Precision: {results['precision']:.4f}")
        self.logger.info(f"  Recall: {results['recall']:.4f}")
        self.logger.info(f"  F1 Score: {results['f1_score']:.4f}")
        
        if 'roc_auc' in results:
            self.logger.info(f"  ROC AUC: {results['roc_auc']:.4f}")
        
        return results
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance scores
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = dict(zip(feature_names, importances))
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            self.logger.info("Top 10 most important features:")
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
                self.logger.info(f"  {i+1}. {feature}: {importance:.4f}")
            
            return feature_importance
        
        else:
            self.logger.warning("Model does not support feature importance")
            return {}
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Best parameters and scores
        """
        training_config = self.config.get_training_config()
        tuning_config = training_config.get('hyperparameter_tuning', {})
        
        if not tuning_config.get('enabled', False):
            self.logger.info("Hyperparameter tuning is disabled")
            return {}
        
        self.logger.info("Starting hyperparameter tuning")
        
        # Define parameter grids for different algorithms
        param_grids = self._get_param_grids()
        
        if self.algorithm not in param_grids:
            self.logger.warning(f"No parameter grid defined for {self.algorithm}")
            return {}
        
        param_grid = param_grids[self.algorithm]
        method = tuning_config.get('method', 'grid_search')
        cv_folds = tuning_config.get('cv_folds', 3)
        
        base_model = self._get_model()
        
        if method == 'grid_search':
            search = GridSearchCV(
                base_model, param_grid,
                cv=cv_folds, scoring='f1_weighted',
                n_jobs=-1, verbose=1
            )
        elif method == 'random_search':
            n_iter = tuning_config.get('n_iter', 50)
            search = RandomizedSearchCV(
                base_model, param_grid,
                n_iter=n_iter, cv=cv_folds,
                scoring='f1_weighted', n_jobs=-1,
                verbose=1, random_state=42
            )
        else:
            raise ValueError(f"Unknown tuning method: {method}")
        
        search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.model = search.best_estimator_
        
        results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }
        
        self.logger.info(f"Best parameters: {search.best_params_}")
        self.logger.info(f"Best CV score: {search.best_score_:.4f}")
        
        return results
    
    def _get_param_grids(self) -> Dict[str, Dict]:
        """Get parameter grids for hyperparameter tuning"""
        return {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [2, 5, 10],
                'max_features': ['sqrt', 'log2']
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
    
    def save_model(self, save_dir: str, model_name: Optional[str] = None) -> str:
        """
        Save trained model to disk
        
        Args:
            save_dir: Directory to save model
            model_name: Optional custom model name
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{self.model_type}_{self.algorithm}_{timestamp}.joblib"
        
        model_path = save_dir / model_name
        
        # Save model with metadata
        model_data = {
            'model': self.model,
            'algorithm': self.algorithm,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'timestamp': datetime.now().isoformat(),
            'feature_columns': getattr(self, 'feature_columns', None)
        }
        
        joblib.dump(model_data, model_path)
        
        self.logger.info(f"Model saved to: {model_path}")
        
        return str(model_path)
    
    def load_model(self, model_path: str):
        """
        Load trained model from disk
        
        Args:
            model_path: Path to saved model
        """
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.algorithm = model_data.get('algorithm', 'unknown')
        self.model_type = model_data.get('model_type', 'unknown')
        self.model_params = model_data.get('model_params', {})
        
        self.logger.info(f"Model loaded from: {model_path}")
        self.logger.info(f"Algorithm: {self.algorithm}, Type: {self.model_type}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using trained model
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Features for prediction
            
        Returns:
            Prediction probabilities array
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded before making predictions")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
        
        return self.model.predict_proba(X)


class ModelManager:
    """Manages multiple models and provides unified interface"""
    
    def __init__(self):
        """Initialize model manager"""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.loan_outcome_model = None
        self.denial_reason_model = None
        
    def train_all_models(self, data_processor, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train both loan outcome and denial reason models
        
        Args:
            data_processor: Data processor instance
            X_train: Training features
            y_train: Training targets (loan outcomes)
            X_test: Test features  
            y_test: Test targets (loan outcomes)
            
        Returns:
            Training results for both models
        """
        results = {}
        
        # Train loan outcome model
        self.logger.info("Training loan outcome prediction model")
        self.loan_outcome_model = ModelTrainer('loan_outcome_model')
        
        loan_train_results = self.loan_outcome_model.train(X_train, y_train)
        loan_eval_results = self.loan_outcome_model.evaluate(X_test, y_test)
        
        results['loan_outcome'] = {
            'training': loan_train_results,
            'evaluation': loan_eval_results,
            'feature_importance': self.loan_outcome_model.get_feature_importance(X_train.columns.tolist())
        }
        
        # Train denial reason model (only on denied loans)
        denied_mask = y_train == 3  # 3 = denied
        if denied_mask.sum() > 50:  # Need at least 50 denied loans for training
            self.logger.info("Training denial reason prediction model")
            
            # Get the original processed data to access denial reasons
            original_data = data_processor.load_data()
            original_data = data_processor.validate_data(original_data)
            original_data = data_processor.create_features(original_data)
            
            # Filter for denied loans and get denial reasons
            denied_data = original_data[original_data['action_taken'] == 3].copy()
            
            # Remove NaN denial reasons
            denied_data = denied_data.dropna(subset=['denial_reason'])
            
            if len(denied_data) > 50 and 'denial_reason' in denied_data.columns:
                self.logger.info(f"Training on {len(denied_data)} denied loans with valid denial reasons")
                
                # Prepare denial reason data
                X_denial, y_denial = data_processor.prepare_features(denied_data, 'denial_reason', is_training=True)
                
                # Check if we have enough data and variety
                if len(y_denial.unique()) > 1:
                    # Split denial reason data
                    from sklearn.model_selection import train_test_split
                    X_denial_train, X_denial_test, y_denial_train, y_denial_test = train_test_split(
                        X_denial, y_denial, test_size=0.2, random_state=42, stratify=y_denial
                    )
                    
                    # Scale features (create a separate scaler for denial reasons)
                    X_denial_train, X_denial_test = data_processor.scale_features(X_denial_train, X_denial_test)
                    
                    # Train denial reason model
                    self.denial_reason_model = ModelTrainer('denial_reason_model')
                    denial_train_results = self.denial_reason_model.train(X_denial_train, y_denial_train)
                    denial_eval_results = self.denial_reason_model.evaluate(X_denial_test, y_denial_test)
                    
                    results['denial_reason'] = {
                        'training': denial_train_results,
                        'evaluation': denial_eval_results,
                        'feature_importance': self.denial_reason_model.get_feature_importance(X_denial_train.columns.tolist())
                    }
                else:
                    self.logger.warning("Not enough variety in denial reasons for training")
            else:
                self.logger.warning("Not enough denied loans with valid denial reasons for training")
        else:
            self.logger.warning(f"Only {denied_mask.sum()} denied loans in training set - skipping denial reason model")
        
        return results
    
    def save_all_models(self, save_dir: str) -> Dict[str, str]:
        """
        Save all trained models
        
        Args:
            save_dir: Directory to save models
            
        Returns:
            Dictionary of model paths
        """
        saved_paths = {}
        
        if self.loan_outcome_model:
            loan_path = self.loan_outcome_model.save_model(save_dir, 'loan_outcome_model.joblib')
            saved_paths['loan_outcome'] = loan_path
        
        if self.denial_reason_model:
            denial_path = self.denial_reason_model.save_model(save_dir, 'denial_reason_model.joblib')
            saved_paths['denial_reason'] = denial_path
        
        return saved_paths
    
    def load_all_models(self, model_dir: str):
        """
        Load all models from directory
        
        Args:
            model_dir: Directory containing saved models
        """
        model_dir = Path(model_dir)
        
        # Load loan outcome model
        loan_model_path = model_dir / 'loan_outcome_model.joblib'
        if loan_model_path.exists():
            self.loan_outcome_model = ModelTrainer()
            self.loan_outcome_model.load_model(loan_model_path)
        
        # Load denial reason model
        denial_model_path = model_dir / 'denial_reason_model.joblib'
        if denial_model_path.exists():
            self.denial_reason_model = ModelTrainer()
            self.denial_reason_model.load_model(denial_model_path)
    
    def predict_loan_outcome(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict loan outcome and probabilities
        
        Args:
            X: Features for prediction
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.loan_outcome_model is None:
            raise ValueError("Loan outcome model not loaded")
        
        predictions = self.loan_outcome_model.predict(X)
        probabilities = self.loan_outcome_model.predict_proba(X)
        
        return predictions, probabilities
    
    def predict_denial_reason(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict denial reason for denied loans
        
        Args:
            X: Features for prediction
            
        Returns:
            Denial reason predictions
        """
        if self.denial_reason_model is None:
            raise ValueError("Denial reason model not loaded")
        
        return self.denial_reason_model.predict(X)