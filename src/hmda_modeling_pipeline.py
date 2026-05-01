#!/usr/bin/env python3
"""
Advanced HMDA Modeling Pipeline with Class Imbalance Handling

This module provides comprehensive machine learning pipeline specifically designed for 
HMDA loan outcome prediction, addressing key challenges:
1. Severe class imbalance in approval vs denial rates
2. Multiple model comparison with proper validation
3. Fair lending compliance monitoring
4. Interpretable predictions with confidence scoring
5. Denial reason prediction capabilities
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, f1_score, precision_recall_curve, roc_curve,
    average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight

# Imbalanced Learning
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

# Advanced Models
import xgboost as xgb
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from hmda_feature_engineer import HMDAFeatureEngineer

# Note: Logging is configured in the main pipeline module
logger = logging.getLogger(__name__)


class HMDAModelingPipeline:
    """Advanced modeling pipeline for HMDA loan prediction with bias handling"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.scaler = RobustScaler()
        self.feature_engineer = HMDAFeatureEngineer()
        self.feature_columns = []
        self.model_performance = {}
        
        # Configure model parameters
        self.model_configs = self._get_model_configurations()
        
    def _get_model_configurations(self) -> Dict[str, Dict]:
        """Get model configurations with hyperparameter grids"""
        
        return {
            'logistic_regression': {
                'model': LogisticRegression(
                    random_state=self.random_state, 
                    max_iter=1000,
                    solver='liblinear'
                ),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'class_weight': ['balanced', None]
                },
                'scoring': 'f1'
            },
            
            'random_forest': {
                'model': RandomForestClassifier(
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', 'balanced_subsample', None],
                    'max_features': ['sqrt', 'log2', None]
                },
                'scoring': 'f1'
            },
            
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    random_state=self.random_state
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.05, 0.1, 0.15, 0.2],
                    'max_depth': [3, 5, 7, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'scoring': 'f1'
            },
            
            'xgboost': {
                'model': xgb.XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss',
                    verbosity=0
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'min_child_weight': [1, 3, 5],
                    'gamma': [0, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'scale_pos_weight': [1, 3, 5, 10]  # For imbalanced data
                },
                'scoring': 'f1'
            },
            
            'lightgbm': {
                'model': lgb.LGBMClassifier(
                    random_state=self.random_state,
                    verbosity=-1
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [31, 63, 127],
                    'min_child_samples': [10, 20, 30],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'class_weight': ['balanced', None]
                },
                'scoring': 'f1'
            }
        }
    
    def handle_class_imbalance(self, X: np.ndarray, y: np.ndarray, 
                             method: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance using various resampling techniques
        
        Methods:
        - smote: SMOTE oversampling
        - borderline_smote: Borderline SMOTE
        - adasyn: ADASYN oversampling
        - smote_tomek: SMOTE + Tomek undersampling
        - smote_enn: SMOTE + ENN undersampling
        - undersample: Random undersampling
        - combined: Custom combination
        """
        
        original_distribution = np.bincount(y)
        logger.info(f"Original class distribution: {original_distribution}")
        
        if method == 'smote':
            sampler = SMOTE(random_state=self.random_state, k_neighbors=3)
        elif method == 'borderline_smote':
            sampler = BorderlineSMOTE(random_state=self.random_state, k_neighbors=3)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=self.random_state, n_neighbors=3)
        elif method == 'smote_tomek':
            sampler = SMOTETomek(random_state=self.random_state)
        elif method == 'smote_enn':
            sampler = SMOTEENN(random_state=self.random_state)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=self.random_state)
        elif method == 'combined':
            # First oversample minority class to 30% of majority
            over = SMOTE(sampling_strategy=0.3, random_state=self.random_state)
            # Then undersample majority to 70% of original
            under = RandomUnderSampler(sampling_strategy=0.7, random_state=self.random_state)
            X_resampled, y_resampled = over.fit_resample(X, y)
            X_resampled, y_resampled = under.fit_resample(X_resampled, y_resampled)
            
            balanced_distribution = np.bincount(y_resampled)
            logger.info(f"Balanced class distribution: {balanced_distribution}")
            return X_resampled, y_resampled
        else:
            return X, y
        
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            balanced_distribution = np.bincount(y_resampled)
            logger.info(f"Balanced class distribution: {balanced_distribution}")
            return X_resampled, y_resampled
        except Exception as e:
            logger.warning(f"Resampling failed with {method}: {str(e)}. Using original data.")
            return X, y
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Prepare data for modeling"""
        
        # Apply feature engineering using the full pipeline to get targets
        logger.info("Applying feature engineering...")
        df_engineered = self.feature_engineer.engineer_all_features(df)
        
        # Filter to clear approve/deny cases only
        df_modeling = df_engineered[df_engineered['target'].isin([0, 1])].copy()
        
        logger.info(f"Dataset size after filtering: {df_modeling.shape}")
        logger.info(f"Target distribution: {df_modeling['target'].value_counts().to_dict()}")
        
        # Prepare features for modeling using the full preparation pipeline
        df_clean, self.feature_columns = self.feature_engineer.prepare_for_modeling(df_modeling)
        
        # Extract features and target
        X = df_clean[self.feature_columns].values
        y = df_clean['target'].values
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Number of features: {len(self.feature_columns)}")
        
        return df_clean, X, y
    
    def train_and_evaluate_models(self, X: np.ndarray, y: np.ndarray, 
                                balance_method: str = 'smote',
                                test_size: float = 0.2,
                                cv_folds: int = 5) -> Dict[str, Any]:
        """Train and evaluate multiple models with hyperparameter tuning"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Skip class imbalance handling to preserve natural denial patterns
        logger.info("Training without class balancing to preserve natural denial patterns")
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
        
        logger.info(f"Using original class distribution: [deny={np.sum(y_train == 0)} approve={np.sum(y_train == 1)}]")
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Train models
        for model_name, config in self.model_configs.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=cv,
                    scoring=config['scoring'],
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train_balanced, y_train_balanced)
                
                # Predictions on test set
                y_pred = grid_search.predict(X_test_scaled)
                y_pred_proba = grid_search.predict_proba(X_test_scaled)[:, 1]
                
                # Calculate metrics
                metrics = {
                    'best_params': grid_search.best_params_,
                    'cv_score': grid_search.best_score_,
                    'test_accuracy': accuracy_score(y_test, y_pred),
                    'test_f1': f1_score(y_test, y_pred),
                    'test_auc': roc_auc_score(y_test, y_pred_proba),
                    'test_precision': classification_report(y_test, y_pred, output_dict=True)['1']['precision'],
                    'test_recall': classification_report(y_test, y_pred, output_dict=True)['1']['recall'],
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                self.models[model_name] = {
                    'model': grid_search.best_estimator_,
                    'grid_search': grid_search,
                    **metrics
                }
                
                logger.info(f"{model_name} - F1: {metrics['test_f1']:.4f}, AUC: {metrics['test_auc']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                continue
        
        # Select best model based on F1 score
        if self.models:
            best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['test_f1'])
            self.best_model = self.models[best_model_name]['model']
            logger.info(f"Best model: {best_model_name} (F1: {self.models[best_model_name]['test_f1']:.4f})")
        
        return {
            'X_test': X_test_scaled,
            'y_test': y_test,
            'models': self.models
        }
    
    def evaluate_fairness(self, df: pd.DataFrame, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model fairness across demographic groups"""
        
        if not self.best_model:
            logger.warning("No trained model available for fairness evaluation")
            return
        
        logger.info("Evaluating model fairness...")
        
        # Get test set indices for demographic analysis
        test_indices = df[df['target'].isin([0, 1])].index[-len(y_test):]
        df_test = df.loc[test_indices].copy()
        
        # Predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        df_test['predicted'] = y_pred
        df_test['predicted_proba'] = y_pred_proba
        
        # Analyze by demographic groups
        demographic_cols = ['applicant_race-1', 'applicant_ethnicity', 'applicant_sex']
        
        fairness_results = {}
        
        for col in demographic_cols:
            if col in df_test.columns:
                group_stats = df_test.groupby(col).agg({
                    'target': ['count', 'mean'],
                    'predicted': 'mean',
                    'predicted_proba': 'mean'
                }).round(4)
                
                fairness_results[col] = group_stats
                
                print(f"\nFairness Analysis - {col}:")
                print(group_stats)
        
        return fairness_results
    
    def plot_model_performance(self):
        """Create comprehensive performance visualizations"""
        
        if not self.models:
            logger.warning("No models to plot")
            return
        
        # Model comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Performance metrics comparison
        metrics = ['test_f1', 'test_auc', 'test_accuracy', 'test_precision', 'test_recall']
        model_names = list(self.models.keys())
        
        for i, metric in enumerate(metrics[:5]):
            if i < 5:
                row, col = i // 3, i % 3
                values = [self.models[model][metric] for model in model_names]
                
                axes[row, col].bar(model_names, values)
                axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
                axes[row, col].set_ylabel('Score')
                axes[row, col].tick_params(axis='x', rotation=45)
                axes[row, col].set_ylim(0, 1)
        
        # Feature importance (if available)
        if self.best_model and hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]  # Top 15 features
            
            axes[1, 2].bar(range(15), importances[indices])
            axes[1, 2].set_title('Top 15 Feature Importances (Best Model)')
            axes[1, 2].set_xlabel('Features')
            axes[1, 2].set_ylabel('Importance')
            
            # Set feature names as x-tick labels
            feature_names = [self.feature_columns[i] for i in indices]
            axes[1, 2].set_xticks(range(15))
            axes[1, 2].set_xticklabels(feature_names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
        
        # Confusion matrices
        if len(self.models) > 0:
            n_models = len(self.models)
            cols = min(3, n_models)
            rows = (n_models + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, (model_name, results) in enumerate(self.models.items()):
                row, col = i // cols, i % cols
                
                cm = results['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col])
                axes[row, col].set_title(f'{model_name} - Confusion Matrix')
                axes[row, col].set_xlabel('Predicted')
                axes[row, col].set_ylabel('Actual')
            
            # Hide empty subplots
            for i in range(n_models, rows * cols):
                row, col = i // cols, i % cols
                axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.show()
    
    def predict_loan_outcome(self, loan_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict loan outcome for new application
        
        Args:
            loan_data: Dictionary containing loan application data
            
        Returns:
            Dictionary with prediction results
        """
        
        if not self.best_model:
            raise ValueError("No trained model available. Train a model first.")
    def predict_loan_outcome(self, loan_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict loan outcome for new application
        
        Args:
            loan_data: Dictionary containing loan application data
            
        Returns:
            Dictionary with prediction results
        """
        
        if not self.best_model:
            raise ValueError("No trained model available. Train a model first.")
        
        # Convert to DataFrame
        df_input = pd.DataFrame([loan_data])
        
        # For prediction, we need to provide default values for missing HMDA columns
        # Get a sample of the training data structure
        try:
            # Load training data to get column structure
            import os
            training_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'actual', 'state_KS.csv')
            df_sample = pd.read_csv(training_path, nrows=1)
            
            # Add missing columns with appropriate defaults
            for col in df_sample.columns:
                if col not in df_input.columns:
                    # Provide reasonable defaults based on column type
                    if col in ['activity_year', 'county_code', 'census_tract', 'state_code']:
                        df_input[col] = df_sample[col].iloc[0]  # Use sample value
                    elif 'action_taken' in col:
                        df_input[col] = 1  # Default to approved for feature engineering
                    elif col.startswith('applicant_') or col.startswith('co-applicant_'):
                        df_input[col] = df_sample[col].iloc[0] if not pd.isna(df_sample[col].iloc[0]) else ''
                    elif col in ['loan_type', 'loan_purpose', 'lien_status']:
                        df_input[col] = 1  # Common defaults
                    elif 'rate' in col.lower() or 'ratio' in col.lower():
                        df_input[col] = df_sample[col].iloc[0] if not pd.isna(df_sample[col].iloc[0]) else 0
                    else:
                        df_input[col] = df_sample[col].iloc[0] if not pd.isna(df_sample[col].iloc[0]) else ''
            
        except Exception as e:
            logger.warning(f"Could not load training data structure: {e}")
            # Fall back to minimal required columns
            df_input['action_taken'] = 1
        
        # Apply the SAME feature engineering pipeline as training
        df_engineered = self.feature_engineer.engineer_all_features(df_input)
        
        # Apply the SAME preparation as training
        df_clean, _ = self.feature_engineer.prepare_for_modeling(df_engineered)
        
        # Extract features in the same order as training
        # Handle case where some features might be missing
        X_input = np.zeros((1, len(self.feature_columns)))
        
        for i, col in enumerate(self.feature_columns):
            if col in df_clean.columns:
                value = df_clean[col].iloc[0]
                # Ensure numeric value
                try:
                    value = float(value)
                    # Ensure finite value
                    if pd.isna(value) or not np.isfinite(value):
                        value = 0.0
                except (ValueError, TypeError):
                    value = 0.0
                X_input[0, i] = value
        
        X_input_scaled = self.scaler.transform(X_input)
        
        # Make prediction
        prediction = self.best_model.predict(X_input_scaled)[0]
        probability = self.best_model.predict_proba(X_input_scaled)[0, 1]
        
        # Risk assessment
        if probability >= 0.8:
            risk_level = 'Low Risk'
        elif probability >= 0.6:
            risk_level = 'Medium Risk'
        elif probability >= 0.4:
            risk_level = 'High Risk'
        else:
            risk_level = 'Very High Risk'
        
        # Confidence score
        confidence = max(probability, 1 - probability)
        
        return {
            'prediction': 'Approved' if prediction == 1 else 'Denied',
            'approval_probability': probability,
            'risk_level': risk_level,
            'confidence': confidence,
            'recommendation': self._get_recommendation(probability, loan_data)
        }
    
    def _get_recommendation(self, probability: float, loan_data: Dict) -> str:
        """Generate recommendation based on prediction"""
        
        if probability >= 0.8:
            return "Strong approval candidate. Proceed with standard processing."
        elif probability >= 0.6:
            return "Good approval candidate. Consider manual review for optimization."
        elif probability >= 0.4:
            return "Borderline case. Detailed manual review recommended."
        else:
            recommendations = []
            
            # Specific recommendations based on risk factors
            if loan_data.get('debt_to_income_ratio', 0) > 43:
                recommendations.append("High DTI ratio - consider debt consolidation options")
            
            if loan_data.get('loan_to_value_ratio', 0) > 0.95:
                recommendations.append("High LTV ratio - larger down payment may improve approval chances")
            
            if not recommendations:
                recommendations.append("High risk application - consider alternative loan products")
            
            return "; ".join(recommendations)
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all trained models"""
        
        if not self.models:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, results in self.models.items():
            summary_data.append({
                'Model': model_name,
                'CV_F1_Score': results['cv_score'],
                'Test_F1_Score': results['test_f1'],
                'Test_AUC_Score': results['test_auc'],
                'Test_Accuracy': results['test_accuracy'],
                'Test_Precision': results['test_precision'],
                'Test_Recall': results['test_recall']
            })
        
        return pd.DataFrame(summary_data).round(4)