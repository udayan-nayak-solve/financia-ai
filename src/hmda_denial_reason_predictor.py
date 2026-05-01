#!/usr/bin/env python3
"""
HMDA Denial Reason Prediction System

This module provides specialized prediction capabilities for HMDA denial reasons,
enabling compliance with fair lending requirements and providing actionable feedback
to loan applicants.

HMDA Denial Reason Codes:
1 - Debt-to-income ratio
2 - Employment history  
3 - Credit history
4 - Collateral
5 - Insufficient cash (downpayment, closing costs)
6 - Unverifiable information
7 - Credit application incomplete
8 - Mortgage insurance denied
9 - Other
10 - Not applicable
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix, hamming_loss
from imblearn.over_sampling import SMOTE

# Note: Logging is configured in the main pipeline module
logger = logging.getLogger(__name__)


class HMDADenialReasonPredictor:
    """Specialized predictor for HMDA denial reasons with multi-label classification"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # Denial reason mappings
        self.denial_reasons = {
            1: 'debt_to_income_ratio',
            2: 'employment_history',
            3: 'credit_history', 
            4: 'collateral',
            5: 'insufficient_cash',
            6: 'unverifiable_information',
            7: 'incomplete_application',
            8: 'mortgage_insurance_denied',
            9: 'other',
            10: 'not_applicable'
        }
        
        # Risk factor mapping to denial reasons
        self.risk_factor_mapping = {
            'debt_to_income_ratio': [1],  # DTI issues
            'loan_to_income_ratio': [1],  # DTI related
            'financial_risk_score': [1, 4],  # DTI and collateral
            'credit_score_proxy': [3],  # Credit history
            'down_payment_ratio': [5],  # Insufficient cash
            'loan_to_value_calculated': [4, 5],  # Collateral and cash
            'property_value': [4],  # Collateral
            'income': [2],  # Employment/income
            'employment_years': [2],  # Employment history
        }
    
    def prepare_denial_data(self, df: pd.DataFrame, feature_columns: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame]]:
        """Prepare data specifically for denial reason prediction"""
        
        # Filter to denied applications only
        denied_data = df[df['action_taken'] == 3].copy()
        
        if denied_data.empty:
            logger.warning("No denied applications found in data")
            return None, None, None
        
        logger.info(f"Found {len(denied_data)} denied applications")
        
        # Find denial reason columns
        denial_cols = [col for col in df.columns if 'denial_reason' in col.lower()]
        
        if not denial_cols:
            logger.warning("No denial reason columns found")
            return None, None, None
        
        # Create binary targets for each denial reason
        denial_targets = np.zeros((len(denied_data), len(self.denial_reasons)))
        
        for i, reason_code in enumerate(self.denial_reasons.keys()):
            # Check if this reason appears in any denial reason column
            reason_mask = (denied_data[denial_cols] == reason_code).any(axis=1)
            denial_targets[:, i] = reason_mask.astype(int)
        
        # Remove samples with no denial reasons or only "not applicable"
        valid_samples = (denial_targets[:, :-1].sum(axis=1) > 0)  # Exclude "not applicable" column
        
        if valid_samples.sum() == 0:
            logger.warning("No valid denial reasons found")
            return None, None, None
        
        denied_data_valid = denied_data[valid_samples].copy()
        denial_targets_valid = denial_targets[valid_samples]
        
        # Extract features
        available_features = [col for col in feature_columns if col in denied_data_valid.columns]
        X_denial = denied_data_valid[available_features].values
        
        logger.info(f"Denial reason data shape: {X_denial.shape}")
        logger.info(f"Denial reason distribution:")
        for i, (code, name) in enumerate(self.denial_reasons.items()):
            count = denial_targets_valid[:, i].sum()
            logger.info(f"  {code} - {name}: {count} cases ({count/len(denial_targets_valid)*100:.1f}%)")
        
        return X_denial, denial_targets_valid, denied_data_valid
    
    def train_denial_reason_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train models for each denial reason"""
        
        if X is None or y is None:
            logger.warning("No data available for training denial reason models")
            return {}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.random_state
        )
        
        results = {}
        
        # Train individual models for each denial reason
        for i, (reason_code, reason_name) in enumerate(self.denial_reasons.items()):
            
            y_reason_train = y_train[:, i]
            y_reason_test = y_test[:, i]
            
            # Skip if too few positive examples
            if y_reason_train.sum() < 5:
                logger.info(f"Skipping {reason_name} - insufficient positive examples ({y_reason_train.sum()})")
                continue
            
            logger.info(f"Training model for {reason_name} (code {reason_code})")
            
            try:
                # Handle class imbalance with SMOTE
                smote = SMOTE(random_state=self.random_state, k_neighbors=min(3, y_reason_train.sum()-1))
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_reason_train)
                
                # Train Random Forest model
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                model.fit(X_train_balanced, y_train_balanced)
                
                # Evaluate
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) > 1 else np.zeros(len(X_test))
                
                # Calculate metrics
                if len(np.unique(y_reason_test)) > 1:  # Only if both classes present in test
                    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
                    
                    f1 = f1_score(y_reason_test, y_pred)
                    precision = precision_score(y_reason_test, y_pred)
                    recall = recall_score(y_reason_test, y_pred)
                    
                    if len(model.classes_) > 1:
                        auc = roc_auc_score(y_reason_test, y_pred_proba)
                    else:
                        auc = 0.0
                else:
                    f1 = precision = recall = auc = 0.0
                
                results[reason_name] = {
                    'model': model,
                    'reason_code': reason_code,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'auc_score': auc,
                    'feature_importance': model.feature_importances_,
                    'positive_cases': y_reason_train.sum(),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logger.info(f"  {reason_name} - F1: {f1:.3f}, AUC: {auc:.3f}, Positive cases: {y_reason_train.sum()}")
                
            except Exception as e:
                logger.error(f"Failed to train model for {reason_name}: {str(e)}")
                continue
        
        # Train multi-output model for combined prediction
        if len(results) > 1:
            logger.info("Training multi-output model for combined denial reason prediction")
            
            try:
                # Use only reasons that we successfully trained individual models for
                successful_reasons = list(results.keys())
                reason_indices = [i for i, (_, name) in enumerate(self.denial_reasons.items()) if name in successful_reasons]
                
                y_multi_train = y_train[:, reason_indices]
                y_multi_test = y_test[:, reason_indices]
                
                # Multi-output classifier
                multi_model = MultiOutputClassifier(
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        class_weight='balanced',
                        random_state=self.random_state,
                        n_jobs=-1
                    )
                )
                
                multi_model.fit(X_train, y_multi_train)
                y_multi_pred = multi_model.predict(X_test)
                
                # Calculate multi-label metrics
                hamming = hamming_loss(y_multi_test, y_multi_pred)
                
                results['multi_output'] = {
                    'model': multi_model,
                    'hamming_loss': hamming,
                    'reason_names': successful_reasons,
                    'reason_indices': reason_indices
                }
                
                logger.info(f"Multi-output model - Hamming loss: {hamming:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train multi-output model: {str(e)}")
        
        self.models = results
        return results
    
    def predict_denial_reasons(self, loan_data: Dict[str, Any], feature_columns: List[str]) -> Dict[str, Any]:
        """Predict denial reasons for a loan application"""
        
        if not self.models:
            raise ValueError("No trained denial reason models available")
        
        # Convert to DataFrame and prepare features
        df_input = pd.DataFrame([loan_data])
        available_features = [col for col in feature_columns if col in df_input.columns]
        
        # Fill missing features with defaults
        for col in feature_columns:
            if col not in df_input.columns:
                df_input[col] = 0
        
        X_input = df_input[feature_columns].values
        X_input_scaled = self.scaler.transform(X_input)
        
        # Get predictions from individual models
        predictions = {}
        probabilities = {}
        
        for reason_name, model_info in self.models.items():
            if reason_name == 'multi_output':
                continue
                
            model = model_info['model']
            prob = model.predict_proba(X_input_scaled)[0, 1] if len(model.classes_) > 1 else 0.0
            pred = model.predict(X_input_scaled)[0]
            
            predictions[reason_name] = bool(pred)
            probabilities[reason_name] = float(prob)
        
        # Get likely denial reasons (probability > 0.5)
        likely_reasons = [reason for reason, prob in probabilities.items() if prob > 0.5]
        
        # Generate explanatory text
        explanations = self._generate_denial_explanations(likely_reasons, loan_data)
        
        return {
            'predicted_reasons': likely_reasons,
            'reason_probabilities': probabilities,
            'explanations': explanations,
            'primary_reason': max(probabilities.keys(), key=lambda k: probabilities[k]) if probabilities else None
        }
    
    def _generate_denial_explanations(self, reasons: List[str], loan_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable explanations for denial reasons"""
        
        explanations = {}
        
        for reason in reasons:
            if reason == 'debt_to_income_ratio':
                dti = loan_data.get('debt_to_income_ratio', 0)
                explanations[reason] = f"Debt-to-income ratio of {dti:.1f}% exceeds acceptable limits (typically 43% or less)"
                
            elif reason == 'employment_history':
                explanations[reason] = "Employment history or income verification concerns identified"
                
            elif reason == 'credit_history':
                explanations[reason] = "Credit history indicates higher risk profile"
                
            elif reason == 'collateral':
                ltv = loan_data.get('loan_to_value_ratio', 0)
                explanations[reason] = f"Property valuation or loan-to-value ratio of {ltv*100:.1f}% presents collateral concerns"
                
            elif reason == 'insufficient_cash':
                down_payment = loan_data.get('down_payment_ratio', 0)
                explanations[reason] = f"Down payment of {down_payment*100:.1f}% may be insufficient for this loan type"
                
            elif reason == 'unverifiable_information':
                explanations[reason] = "Unable to verify key information provided in application"
                
            elif reason == 'incomplete_application':
                explanations[reason] = "Application missing required documentation or information"
                
            else:
                explanations[reason] = f"Concerns identified related to {reason.replace('_', ' ')}"
        
        return explanations
    
    def analyze_denial_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze patterns in denial reasons across demographic groups"""
        
        if not self.models:
            logger.warning("No trained models available for pattern analysis")
            return pd.DataFrame()
        
        # Get denied applications
        denied_data = df[df['action_taken'] == 3].copy()
        
        if denied_data.empty:
            return pd.DataFrame()
        
        # Analyze by demographic groups
        analysis_results = []
        
        demographic_cols = ['applicant_race-1', 'applicant_ethnicity', 'applicant_sex']
        denial_cols = [col for col in df.columns if 'denial_reason' in col.lower()]
        
        for demo_col in demographic_cols:
            if demo_col in denied_data.columns:
                
                for demo_value in denied_data[demo_col].unique():
                    if pd.isna(demo_value):
                        continue
                    
                    subset = denied_data[denied_data[demo_col] == demo_value]
                    
                    # Count each denial reason
                    for reason_code, reason_name in self.denial_reasons.items():
                        count = ((subset[denial_cols] == reason_code).any(axis=1)).sum()
                        
                        analysis_results.append({
                            'demographic_group': demo_col,
                            'demographic_value': demo_value,
                            'denial_reason': reason_name,
                            'denial_reason_code': reason_code,
                            'count': count,
                            'percentage': count / len(subset) * 100 if len(subset) > 0 else 0,
                            'total_denials': len(subset)
                        })
        
        return pd.DataFrame(analysis_results)
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of denial reason model performance"""
        
        if not self.models:
            return pd.DataFrame()
        
        summary_data = []
        
        for reason_name, model_info in self.models.items():
            if reason_name == 'multi_output':
                continue
                
            summary_data.append({
                'Denial_Reason': reason_name,
                'Reason_Code': model_info['reason_code'],
                'F1_Score': model_info['f1_score'],
                'Precision': model_info['precision'],
                'Recall': model_info['recall'],
                'AUC_Score': model_info['auc_score'],
                'Positive_Cases': model_info['positive_cases']
            })
        
        return pd.DataFrame(summary_data).round(4)
    
    def explain_model_decisions(self, feature_columns: List[str], top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Get feature importance explanations for each denial reason model"""
        
        explanations = {}
        
        for reason_name, model_info in self.models.items():
            if reason_name == 'multi_output':
                continue
                
            if 'feature_importance' in model_info:
                importances = model_info['feature_importance']
                
                # Get top N most important features
                indices = np.argsort(importances)[::-1][:top_n]
                top_features = [(feature_columns[i], importances[i]) for i in indices]
                
                explanations[reason_name] = top_features
        
        return explanations