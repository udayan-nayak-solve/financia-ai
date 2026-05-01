"""
Enhanced Model Training with Optuna Hyperparameter Optimization
Provides advanced hyperparameter tuning capabilities using Bayesian optimization
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Any, Tuple, List, Optional, Callable
import logging
import joblib
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

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


class OptunaModelTrainer:
    """Enhanced model trainer with Optuna hyperparameter optimization"""
    
    def __init__(self, model_type: str = 'loan_outcome_model'):
        """
        Initialize enhanced model trainer
        
        Args:
            model_type: Type of model to train ('loan_outcome_model' or 'denial_reason_model')
        """
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type
        self.model = None
        self.best_params = None
        self.study = None
        self.algorithm = self.config.get(f'{model_type}.algorithm', 'random_forest')
        
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not available. Falling back to basic hyperparameter tuning.")
    
    def optimize_hyperparameters(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            n_trials: Number of optimization trials
            timeout: Timeout in seconds (optional)
            n_jobs: Number of parallel jobs
            
        Returns:
            Optimization results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for advanced hyperparameter optimization. "
                            "Install with: pip install optuna")
        
        self.logger.info(f"Starting Optuna optimization for {self.algorithm} ({n_trials} trials)")
        
        # Create study
        study_name = f"{self.model_type}_{self.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Configure pruner for early stopping of unpromising trials
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        # Configure sampler for Bayesian optimization
        sampler = TPESampler(seed=42)
        
        self.study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=sampler,
            pruner=pruner
        )
        
        # Define objective function
        def objective(trial):
            return self._objective_function(trial, X_train, y_train, X_val, y_val)
        
        # Optimize
        try:
            self.study.optimize(
                objective, 
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            self.logger.info("Optimization interrupted by user")
        
        # Get best parameters
        self.best_params = self.study.best_params
        
        # Train final model with best parameters
        self.model = self._create_model_with_params(self.best_params)
        
        if X_val is not None:
            # Use validation set for early stopping if available
            if self.algorithm in ['xgboost', 'lightgbm']:
                eval_set = [(X_train, y_train), (X_val, y_val)]
                self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            else:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        # Return optimization results
        results = {
            'best_params': self.best_params,
            'best_score': self.study.best_value,
            'n_trials': len(self.study.trials),
            'study_name': study_name,
            'optimization_history': [trial.value for trial in self.study.trials if trial.value is not None]
        }
        
        self.logger.info(f"Optimization completed: Best score = {self.study.best_value:.4f}")
        self.logger.info(f"Best parameters: {self.best_params}")
        
        return results
    
    def _objective_function(
        self, 
        trial, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> float:
        """
        Objective function for Optuna optimization
        
        Args:
            trial: Optuna trial object
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Objective score to maximize
        """
        try:
            # Get hyperparameters for this trial
            params = self._suggest_hyperparameters(trial)
            
            # Create model with suggested parameters
            model = self._create_model_with_params(params)
            
            # Use validation set if available, otherwise use cross-validation
            if X_val is not None and y_val is not None:
                # Train on training set, evaluate on validation set
                if self.algorithm in ['xgboost', 'lightgbm']:
                    eval_set = [(X_train, y_train), (X_val, y_val)]
                    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
                else:
                    model.fit(X_train, y_train)
                
                # Get predictions
                y_pred = model.predict(X_val)
                
                # Calculate F1 score (weighted for multi-class)
                score = f1_score(y_val, y_pred, average='weighted')
                
            else:
                # Use cross-validation
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=1)
                score = scores.mean()
            
            # Report intermediate value for pruning
            trial.report(score, step=0)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Trial failed: {e}")
            return 0.0
    
    def _suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for the current algorithm
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        if self.algorithm == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 25),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
                'random_state': 42,
                'n_jobs': -1
            }
        
        elif self.algorithm == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available")
            
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 3.0),
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss'
            }
        
        elif self.algorithm == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available")
            
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
        
        elif self.algorithm == 'logistic_regression':
            return {
                'C': trial.suggest_float('C', 0.01, 100.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0) if trial.params.get('penalty') == 'elasticnet' else None,
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _create_model_with_params(self, params: Dict[str, Any]):
        """
        Create model instance with given parameters
        
        Args:
            params: Model parameters
            
        Returns:
            Initialized model instance
        """
        # Remove None values from params
        params = {k: v for k, v in params.items() if v is not None}
        
        if self.algorithm == 'random_forest':
            return RandomForestClassifier(**params)
        
        elif self.algorithm == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available")
            return xgb.XGBClassifier(**params)
        
        elif self.algorithm == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available")
            return lgb.LGBMClassifier(**params)
        
        elif self.algorithm == 'logistic_regression':
            return LogisticRegression(**params)
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def train_with_best_params(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Train model with best parameters from optimization
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training results
        """
        if self.best_params is None:
            raise ValueError("No best parameters available. Run hyperparameter optimization first.")
        
        self.logger.info(f"Training final model with best parameters: {self.best_params}")
        
        # Create model with best parameters
        self.model = self._create_model_with_params(self.best_params)
        
        # Train model
        if self.algorithm in ['xgboost', 'lightgbm'] and X_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X_train, y_train)
        
        return {'best_params_used': self.best_params}
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """
        Get insights from the optimization process
        
        Returns:
            Dictionary containing optimization insights
        """
        if self.study is None:
            return {}
        
        insights = {
            'best_score': self.study.best_value,
            'best_params': self.study.best_params,
            'n_trials': len(self.study.trials),
            'pruned_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'complete_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'failed_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])
        }
        
        # Parameter importance (if available)
        try:
            param_importance = optuna.importance.get_param_importances(self.study)
            insights['param_importance'] = param_importance
        except Exception:
            insights['param_importance'] = {}
        
        return insights
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot optimization history
        
        Args:
            save_path: Path to save plot (optional)
        """
        if not OPTUNA_AVAILABLE or self.study is None:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Plot optimization history
            fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Optimization history plot saved to: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            self.logger.warning(f"Could not create optimization plot: {e}")
    
    def plot_param_importances(self, save_path: Optional[str] = None):
        """
        Plot parameter importances
        
        Args:
            save_path: Path to save plot (optional)
        """
        if not OPTUNA_AVAILABLE or self.study is None:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Plot parameter importances
            fig = optuna.visualization.matplotlib.plot_param_importances(self.study)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Parameter importance plot saved to: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            self.logger.warning(f"Could not create parameter importance plot: {e}")
    
    def save_study(self, save_path: str):
        """
        Save Optuna study for later analysis
        
        Args:
            save_path: Path to save study
        """
        if self.study is None:
            raise ValueError("No study to save")
        
        study_data = {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                for trial in self.study.trials
            ],
            'algorithm': self.algorithm,
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(save_path, 'w') as f:
            json.dump(study_data, f, indent=2)
        
        self.logger.info(f"Study saved to: {save_path}")
    
    # Standard methods from original ModelTrainer
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        self.logger.info("Evaluating model performance")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = None
        
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test)
        
        results = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
            results['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba[:, 1]))
        
        self.logger.info(f"Model Evaluation Results:")
        self.logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        self.logger.info(f"  F1 Score: {results['f1_score']:.4f}")
        
        return results
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = dict(zip(feature_names, importances))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        else:
            self.logger.warning("Model does not support feature importance")
            return {}
    
    def save_model(self, save_dir: str, model_name: Optional[str] = None) -> str:
        """Save trained model and optimization results"""
        if self.model is None:
            raise ValueError("No model to save")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{self.model_type}_{self.algorithm}_optuna_{timestamp}.joblib"
        
        model_path = save_dir / model_name
        
        model_data = {
            'model': self.model,
            'algorithm': self.algorithm,
            'model_type': self.model_type,
            'best_params': self.best_params,
            'optimization_insights': self.get_optimization_insights(),
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        self.logger.info(f"Optimized model saved to: {model_path}")
        
        return str(model_path)