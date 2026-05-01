#!/usr/bin/env python3
"""
Loan Outcome Prediction System

Advanced ML models to predict:
1. Loan approval/denial outcomes
2. Denial reasons when loans are rejected
3. Risk assessment for loan applications

Uses ensemble methods and feature engineering for production-grade predictions.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
import xgboost as xgb

# Configuration
# Note: Logging is configured in the main pipeline module
logger = logging.getLogger(__name__)


class LoanOutcomePredictor:
    """Advanced loan outcome prediction system"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.encoders = {}
        self.scalers = {}
        self.feature_columns = []
        
        # HMDA Denial Reason Codes (based on official HMDA specification)
        self.DENIAL_REASON_CODES = {
            1: "Debt-to-income ratio",
            2: "Employment history",
            3: "Credit history",
            4: "Collateral",
            5: "Insufficient cash (downpayment, closing costs)",
            6: "Unverifiable information",
            7: "Credit application incomplete",
            8: "Mortgage insurance denied",
            9: "Other"
        }
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for loan outcome prediction"""
        
        logger.info("Preparing features for loan outcome prediction...")
        
        # Create a copy for processing
        features_df = df.copy()
        
        # Select relevant features for prediction (prioritized based on importance)
        # MOST IMPORTANT FEATURES (High Priority)
        high_priority_features = [
            'debt_to_income_ratio',   # Critical risk indicator
            'loan_to_value_ratio',    # Critical risk indicator 
            'property_value',         # Collateral value
            'loan_amount',           # Loan size
            'loan_purpose',          # Purpose of loan
            'income'                 # Borrower income
        ]
        
        # LESS IMPORTANT FEATURES (Lower Priority)
        low_priority_features = [
            'derived_ethnicity',      # Demographic factor
            'derived_race',          # Demographic factor
            'derived_sex'            # Demographic factor
        ]
        
        # ADDITIONAL MODEL FEATURES (Technical requirements)
        additional_features = [
            'applicant_credit_score_type', 'loan_type', 'occupancy_type',
            'derived_loan_product_type', 'derived_dwelling_category', 'lien_status'
        ]
        
        # Combine all features in priority order
        feature_cols = high_priority_features + low_priority_features + additional_features
        
        # Filter columns that exist in the dataframe
        available_cols = [col for col in feature_cols if col in features_df.columns]
        
        # For prediction, we don't need target columns
        target_cols = ['action_taken', 'denial_reason-1', 'denial_reason-2', 'denial_reason-3', 'denial_reason-4']
        existing_targets = [col for col in target_cols if col in features_df.columns]
        
        if existing_targets:
            # Training mode - include target columns
            features_df = features_df[available_cols + existing_targets].copy()
        else:
            # Prediction mode - only feature columns
            features_df = features_df[available_cols].copy()
        
        # Handle missing values
        # Numeric features
        numeric_cols = ['loan_amount', 'income', 'debt_to_income_ratio', 
                       'loan_to_value_ratio', 'property_value', 'applicant_credit_score_type']
        
        for col in numeric_cols:
            if col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
                features_df[col] = features_df[col].fillna(features_df[col].median())
        
        # Categorical features - fill missing values
        categorical_cols = ['loan_type', 'loan_purpose', 'occupancy_type', 
                           'derived_loan_product_type', 'derived_dwelling_category',
                           'derived_ethnicity', 'derived_race', 'derived_sex', 'lien_status']
        
        for col in categorical_cols:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna('Unknown')
        
        # Create additional engineered features
        if 'loan_amount' in features_df.columns and 'income' in features_df.columns:
            # Avoid division by zero
            features_df['loan_to_income_ratio'] = features_df['loan_amount'] / np.maximum(features_df['income'], 1)
        
        if 'property_value' in features_df.columns and 'loan_amount' in features_df.columns:
            # Avoid division by zero and ensure reasonable values
            features_df['down_payment_ratio'] = 1 - (features_df['loan_amount'] / np.maximum(features_df['property_value'], 1))
            # Cap extreme values
            features_df['down_payment_ratio'] = np.clip(features_df['down_payment_ratio'], -1, 1)
        
        # Create target variables if action_taken exists (training mode)
        if 'action_taken' in features_df.columns:
            features_df['loan_approved'] = features_df['action_taken'].isin([1, 2, 6, 8]).astype(int)
            features_df['loan_denied'] = (features_df['action_taken'] == 3).astype(int)
        
        # Process denial reasons (multi-label) if they exist
        denial_reason_cols = ['denial_reason-1', 'denial_reason-2', 'denial_reason-3', 'denial_reason-4']
        existing_denial_cols = [col for col in denial_reason_cols if col in features_df.columns]
        
        for col in existing_denial_cols:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(10)
        
        # Define feature columns (exclude target variables)
        exclude_cols = ['action_taken', 'loan_approved', 'loan_denied'] + existing_denial_cols
        self.feature_columns = [col for col in features_df.columns if col not in exclude_cols]
        
        # Handle infinite and extreme values
        for col in self.feature_columns:
            if col in features_df.columns and features_df[col].dtype in ['float64', 'int64']:
                # Replace infinite values with NaN
                features_df[col] = features_df[col].replace([np.inf, -np.inf], np.nan)
                # Fill NaN with median
                features_df[col] = features_df[col].fillna(features_df[col].median())
                # Cap extreme values (beyond 99.9th percentile)
                upper_cap = features_df[col].quantile(0.999)
                lower_cap = features_df[col].quantile(0.001)
                features_df[col] = np.clip(features_df[col], lower_cap, upper_cap)
        
        logger.info(f"Features prepared: {len(self.feature_columns)} features")
        return features_df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit_encoders: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        
        df_encoded = df.copy()
        
        # Identify categorical columns
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col in self.feature_columns]
        
        if fit_encoders:
            self.encoders = {}
        
        for col in categorical_cols:
            if fit_encoders:
                # Use Label Encoder for simplicity
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
            else:
                # Transform using existing encoder
                if col in self.encoders:
                    # Handle unseen categories
                    le = self.encoders[col]
                    unique_vals = set(le.classes_)
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: x if x in unique_vals else 'Unknown'
                    )
                    df_encoded[col] = le.transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def train_approval_model(self, df: pd.DataFrame) -> Dict:
        """Train loan approval prediction model"""
        
        logger.info("Training loan approval prediction model...")
        
        # Prepare features
        features_df = self.prepare_features(df)
        features_encoded = self.encode_categorical_features(features_df, fit_encoders=True)
        
        # Prepare training data
        X = features_encoded[self.feature_columns]
        y = features_encoded['loan_approved']
        
        # Scale features
        self.scalers['approval'] = StandardScaler()
        X_scaled = self.scalers['approval'].fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train ensemble model
        models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'xgb': xgb.XGBClassifier(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = 0
        model_scores = {}
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            avg_score = cv_scores.mean()
            model_scores[name] = avg_score
            
            logger.info(f"{name.upper()} CV ROC-AUC: {avg_score:.4f} (+/- {cv_scores.std()*2:.4f})")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
        
        # Train best model on full training set
        best_model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        test_accuracy = accuracy_score(y_test, y_pred)
        test_roc_auc = roc_auc_score(y_test, y_prob)
        
        logger.info(f"Best model test accuracy: {test_accuracy:.4f}")
        logger.info(f"Best model test ROC-AUC: {test_roc_auc:.4f}")
        
        # Log feature importance to verify prioritization
        self._log_feature_importance(best_model, 'Loan Approval')
        
        # Store model
        self.models['approval'] = best_model
        
        return {
            'model': best_model,
            'accuracy': test_accuracy,
            'roc_auc': test_roc_auc,
            'cv_scores': model_scores,
            'classification_report': classification_report(y_test, y_pred)
        }
    
    def train_denial_reason_model(self, df: pd.DataFrame) -> Dict:
        """Train denial reason prediction model (multi-label)"""
        
        logger.info("Training denial reason prediction model...")
        
        # Filter only denied loans
        denied_loans = df[df['action_taken'] == 3].copy()
        
        if len(denied_loans) < 50:
            logger.warning("Insufficient denied loans for training denial reason model")
            return None
        
        # Prepare features
        features_df = self.prepare_features(denied_loans)
        features_encoded = self.encode_categorical_features(features_df, fit_encoders=False)
        
        # Prepare denial reason targets
        denial_cols = ['denial_reason-1', 'denial_reason-2', 'denial_reason-3', 'denial_reason-4']
        available_denial_cols = [col for col in denial_cols if col in features_encoded.columns]
        
        if not available_denial_cols:
            logger.warning("No denial reason columns available")
            return None
        
        X = features_encoded[self.feature_columns]
        y = features_encoded[available_denial_cols]
        
        # Convert denial reasons to binary matrix
        y_binary = pd.DataFrame()
        for col in available_denial_cols:
            for reason in range(1, 10):  # Reasons 1-9 (10 is NA)
                y_binary[f'reason_{reason}'] = (y[col] == reason).astype(int)
        
        # Remove columns with too few positive examples
        min_samples = max(10, len(y_binary) * 0.01)  # At least 1% or 10 samples
        y_binary = y_binary.loc[:, y_binary.sum() >= min_samples]
        
        if y_binary.empty:
            logger.warning("No denial reasons with sufficient samples")
            return None
        
        # Scale features
        if 'denial' not in self.scalers:
            self.scalers['denial'] = StandardScaler()
            X_scaled = self.scalers['denial'].fit_transform(X)
        else:
            X_scaled = self.scalers['denial'].transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_binary, test_size=0.2, random_state=42
        )
        
        # Train multi-output model
        base_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        multi_model = MultiOutputClassifier(base_model)
        multi_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = multi_model.predict(X_test)
        
        # Calculate accuracy for each denial reason
        reason_accuracies = {}
        for i, col in enumerate(y_binary.columns):
            if y_test.iloc[:, i].sum() > 0:  # Only if there are positive examples
                acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
                reason_accuracies[col] = acc
        
        avg_accuracy = np.mean(list(reason_accuracies.values())) if reason_accuracies else 0
        
        logger.info(f"Denial reason model average accuracy: {avg_accuracy:.4f}")
        
        # Store model
        self.models['denial_reasons'] = multi_model
        self.models['denial_reason_columns'] = y_binary.columns.tolist()
        
        return {
            'model': multi_model,
            'average_accuracy': avg_accuracy,
            'reason_accuracies': reason_accuracies,
            'denial_reason_columns': y_binary.columns.tolist()
        }
    
    def predict_loan_outcome(self, application_data: Dict) -> Dict:
        """Predict loan outcome for new application with full AI transparency"""
        
        if 'approval' not in self.models:
            raise ValueError("Approval model not trained yet")
        
        # Convert application data to DataFrame
        app_df = pd.DataFrame([application_data])
        
        # Prepare features (same as training)
        features_df = self.prepare_features(app_df)
        features_encoded = self.encode_categorical_features(features_df, fit_encoders=False)
        
        # Scale features
        X = features_encoded[self.feature_columns]
        X_scaled = self.scalers['approval'].transform(X)
        
        # Predict approval probability
        approval_prob = self.models['approval'].predict_proba(X_scaled)[0, 1]
        approval_pred = self.models['approval'].predict(X_scaled)[0]
        
        result = {
            'approval_probability': float(approval_prob),
            'predicted_outcome': 'Approved' if approval_pred == 1 else 'Denied',
            'confidence': max(approval_prob, 1 - approval_prob)
        }
        
        # Add AI transparency features
        try:
            # Get feature importance from the model if it's available (XGBoost, RandomForest, etc.)
            if hasattr(self.models['approval'], 'feature_importances_'):
                feature_importance = dict(zip(self.feature_columns, self.models['approval'].feature_importances_))
                # Sort by importance and get top contributing features
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                result['feature_importance'] = dict(sorted_features[:8])  # Top 8 features
                
            # Add model features used in this prediction
            model_features = {}
            for i, col in enumerate(self.feature_columns):
                if i < len(X_scaled[0]):
                    model_features[col] = float(X_scaled[0][i])
            result['model_features_used'] = model_features
            
            # Add input data summary for transparency
            key_inputs = {
                'debt_to_income_ratio': application_data.get('debt_to_income_ratio', 'N/A'),
                'loan_to_value_ratio': application_data.get('loan_to_value_ratio', 'N/A'),
                'property_value': application_data.get('property_value', 'N/A'),
                'loan_amount': application_data.get('loan_amount', 'N/A'),
                'income': application_data.get('income', 'N/A'),
                'applicant_credit_score_type': application_data.get('applicant_credit_score_type', 'N/A')
            }
            result['key_application_inputs'] = key_inputs
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
        
        # If likely to be denied, predict denial reasons using hybrid approach
        if approval_pred == 0:
            # Use both ML model (if available) and intelligent rules
            denial_reasons = self._predict_denial_reasons_hybrid(application_data, X_scaled if 'denial_reasons' in self.models else None)
            result['predicted_denial_reasons'] = denial_reasons
        else:
            # Even for approved loans, if risk factors are extremely high, provide warnings
            risk_reasons = self._predict_denial_reasons_intelligent(application_data)
            if risk_reasons and any(term in ' '.join(risk_reasons).lower() for term in ['exceeds 50%', 'extremely', 'very high']):
                result['risk_warnings'] = risk_reasons[:2]  # Top 2 risk factors as warnings
        
        return result
    
    def _predict_denial_reasons_hybrid(self, application_data: Dict, X_scaled=None) -> List[str]:
        """
        Predict denial reasons using hybrid approach:
        1. Use trained ML model if available (learned from actual HMDA data)
        2. Fall back to intelligent rule-based logic
        3. Prioritize rule-based logic for extreme financial ratios
        """
        
        ml_reasons = []
        rule_reasons = []
        
        # Method 1: Always get rule-based prediction first
        rule_reasons = self._predict_denial_reasons_intelligent(application_data)
        
        # Method 2: Use trained ML model if available
        if X_scaled is not None and 'denial_reasons' in self.models:
            try:
                denial_pred = self.models['denial_reasons'].predict(X_scaled)[0]
                
                # Get top predicted denial reasons from ML model
                for i, pred in enumerate(denial_pred):
                    if pred == 1:
                        reason_col = self.models['denial_reason_columns'][i]
                        reason_num = int(reason_col.split('_')[1])
                        if reason_num in self.DENIAL_REASON_CODES:
                            ml_reasons.append(self.DENIAL_REASON_CODES[reason_num])
            except Exception as e:
                logger.warning(f"ML denial reason prediction failed: {e}")
        
        # Method 3: Intelligent combination based on scenario
        final_reasons = []
        
        # Extract key metrics for decision logic
        debt_to_income = application_data.get('debt_to_income_ratio', 0)
        loan_to_value = application_data.get('loan_to_value_ratio', 0)
        credit_score = application_data.get('applicant_credit_score_type', 0)
        
        # Priority 1: For extreme financial ratios, heavily favor rule-based logic
        if debt_to_income > 50 or loan_to_value > 95:
            # Use rule-based reasons first for extreme cases
            final_reasons.extend(rule_reasons[:2])
            # Only add ML reasons if they don't conflict
            for ml_reason in ml_reasons:
                if ml_reason not in final_reasons and len(final_reasons) < 3:
                    final_reasons.append(f"{ml_reason} (ML confirmed)")
        
        # Priority 2: For moderate ratios, balance both approaches
        elif debt_to_income > 36 or loan_to_value > 85:
            # Combine both, starting with most relevant
            financial_ml = [r for r in ml_reasons if any(term in r.lower() for term in ['debt', 'collateral', 'cash'])]
            final_reasons.extend(financial_ml[:1])  # Top financial ML reason
            
            # Add rule-based if not covered
            for rule_reason in rule_reasons:
                if not any(similar_term in rule_reason.lower() for similar_term in 
                          [word.lower() for word in ' '.join(final_reasons).split()]):
                    final_reasons.append(rule_reason)
                    break
            
            # Add remaining ML reasons
            for ml_reason in ml_reasons:
                if ml_reason not in final_reasons and len(final_reasons) < 3:
                    final_reasons.append(ml_reason)
        
        # Priority 3: For credit-driven denials, favor ML model
        elif credit_score < 620:
            # Credit issues - ML model might be more nuanced
            credit_ml = [r for r in ml_reasons if 'credit' in r.lower()]
            final_reasons.extend(credit_ml[:1])
            
            # Add rule-based credit reason if ML didn't catch it
            credit_rules = [r for r in rule_reasons if 'credit' in r.lower()]
            if not final_reasons and credit_rules:
                final_reasons.extend(credit_rules[:1])
            
            # Add other reasons
            other_reasons = rule_reasons + [r for r in ml_reasons if r not in final_reasons]
            final_reasons.extend(other_reasons[:2])
        
        # Priority 4: Default case - use rule-based primarily
        else:
            final_reasons.extend(rule_reasons[:2])
            # Add ML reasons that aren't redundant
            for ml_reason in ml_reasons:
                if ml_reason not in final_reasons and len(final_reasons) < 3:
                    final_reasons.append(ml_reason)
        
        # Ensure we always have at least one reason
        if not final_reasons:
            final_reasons = rule_reasons if rule_reasons else ml_reasons if ml_reasons else ["Other"]
        
        return final_reasons[:3]  # Return top 3 reasons
    
    def _predict_denial_reasons_intelligent(self, application_data: Dict) -> List[str]:
        """
        Predict denial reasons using intelligent rule-based logic that maps to
        official HMDA denial reason codes (1-9)
        """
        
        denial_reasons = []
        
        # Extract key metrics
        debt_to_income = application_data.get('debt_to_income_ratio', 0)
        loan_to_value = application_data.get('loan_to_value_ratio', 0)
        credit_score = application_data.get('applicant_credit_score_type', 0)
        income = application_data.get('income', 0)
        loan_amount = application_data.get('loan_amount', 0)
        property_value = application_data.get('property_value', 0)
        
        # Rule 1: Debt-to-Income Ratio (HMDA Code 1)
        if debt_to_income > 43:  # Standard DTI threshold
            if debt_to_income > 50:
                denial_reasons.append("Debt-to-income ratio (exceeds 50%)")
            else:
                denial_reasons.append("Debt-to-income ratio")
        
        # Rule 2: Credit History (HMDA Code 3)
        if credit_score < 620:  # Poor credit
            denial_reasons.append("Credit history")
        elif credit_score < 680 and debt_to_income > 40:
            denial_reasons.append("Credit history (marginal score with high DTI)")
        
        # Rule 3: Collateral (HMDA Code 4) - Property value / LTV issues
        if loan_to_value > 95:  # High LTV threshold
            denial_reasons.append("Collateral (high loan-to-value ratio)")
        elif property_value > 0 and loan_amount > property_value:
            denial_reasons.append("Collateral (loan exceeds property value)")
        
        # Rule 4: Insufficient Cash - Down Payment (HMDA Code 5)
        if loan_to_value > 90 and debt_to_income > 36:  # High LTV + DTI
            denial_reasons.append("Insufficient cash (downpayment, closing costs)")
        elif loan_to_value > 97:  # Very high LTV
            denial_reasons.append("Insufficient cash (downpayment, closing costs)")
        
        # Rule 5: Employment History (HMDA Code 2) - Income related
        if income < 30000:  # Very low income
            denial_reasons.append("Employment history (insufficient income)")
        elif income > 0 and loan_amount > 0:
            loan_to_income = loan_amount / income
            if loan_to_income > 5:  # Loan amount is more than 5x annual income
                denial_reasons.append("Employment history (income insufficient for loan size)")
        
        # Rule 6: Multiple Risk Factors Leading to "Other" (HMDA Code 9)
        risk_factors = 0
        if debt_to_income > 36:
            risk_factors += 1
        if loan_to_value > 80:
            risk_factors += 1
        if credit_score < 700:
            risk_factors += 1
        
        if risk_factors >= 2 and len(denial_reasons) == 0:
            denial_reasons.append("Other (multiple risk factors)")
        
        # Rule 7: Conservative Lending Standards
        if len(denial_reasons) == 0:
            if debt_to_income > 28:
                denial_reasons.append("Debt-to-income ratio (exceeds conservative guidelines)")
            elif loan_to_value > 85:
                denial_reasons.append("Insufficient cash (downpayment, closing costs)")
            elif credit_score < 740:
                denial_reasons.append("Credit history")
            else:
                denial_reasons.append("Other")
        
        return denial_reasons[:3]  # Return top 3 reasons to avoid overwhelming output
    
    def _log_feature_importance(self, model, model_name: str):
        """Log feature importance to verify key features are prioritized"""
        
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info(f"\n🎯 {model_name} Model - Feature Importance Analysis:")
            logger.info("="*60)
            
            # Define your priority features
            high_priority = ['debt_to_income_ratio', 'loan_to_value_ratio', 'property_value', 
                           'loan_amount', 'loan_purpose', 'income']
            low_priority = ['derived_ethnicity', 'derived_race', 'derived_sex']
            
            logger.info("📊 TOP 10 MOST IMPORTANT FEATURES:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
                priority = "🔴 HIGH" if row['feature'] in high_priority else "🟡 LOW" if row['feature'] in low_priority else "⚪ OTHER"
                logger.info(f"   {i:2d}. {row['feature']:<25} : {row['importance']:.4f} ({priority})")
            
            # Summary of your key features
            logger.info("\n🎯 YOUR KEY FEATURES RANKING:")
            for feature in high_priority + low_priority:
                if feature in feature_importance['feature'].values:
                    rank = feature_importance[feature_importance['feature'] == feature].index[0] + 1
                    importance = feature_importance[feature_importance['feature'] == feature]['importance'].iloc[0]
                    priority = "HIGH" if feature in high_priority else "LOW"
                    logger.info(f"   {feature:<25}: Rank #{rank:2d}, Importance: {importance:.4f} (Priority: {priority})")
            
            logger.info("="*60)
    
    def save_models(self, filepath: Path):
        """Save trained models"""
        
        model_data = {
            'models': self.models,
            'encoders': self.encoders,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: Path):
        """Load trained models"""
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.encoders = model_data['encoders']
        self.scalers = model_data['scalers']
        self.feature_columns = model_data['feature_columns']
        
        logger.info(f"Models loaded from {filepath}")


def train_loan_prediction_models(data_path: Path, config) -> LoanOutcomePredictor:
    """Train all loan prediction models"""
    
    logger.info("Starting loan prediction model training...")
    
    # Load HMDA data
    hmda_data = pd.read_csv(data_path)
    
    # Initialize predictor
    predictor = LoanOutcomePredictor(config)
    
    # Train approval model
    approval_results = predictor.train_approval_model(hmda_data)
    
    # Train denial reason model
    denial_results = predictor.train_denial_reason_model(hmda_data)
    
    # Save models
    model_path = config.MODELS_DIR / "loan_outcome_models.pkl"
    predictor.save_models(model_path)
    
    logger.info("Loan prediction models training completed")
    
    return predictor


if __name__ == "__main__":
    from advanced_lending_platform import LendingConfig
    
    # Initialize configuration
    config = LendingConfig()
    
    # Train models
    hmda_path = config.DATA_DIR / "state_KS.csv"
    predictor = train_loan_prediction_models(hmda_path, config)
    
    # Example prediction
    sample_application = {
        'loan_amount': 250000,
        'income': 75000,
        'debt_to_income_ratio': 28,
        'loan_to_value_ratio': 80,
        'property_value': 312500,
        'applicant_credit_score_type': 3,
        'loan_type': 1,
        'loan_purpose': 1,
        'occupancy_type': 1,
        'derived_loan_product_type': 'Conventional:First Lien',
        'derived_dwelling_category': 'Single Family (1-4 Units):Site-Built',
        'derived_ethnicity': 'Not Hispanic or Latino',
        'derived_race': 'White',
        'derived_sex': 'Male',
        'lien_status': 1
    }
    
    result = predictor.predict_loan_outcome(sample_application)
    print("\nSample Loan Prediction:")
    print(f"Outcome: {result['predicted_outcome']}")
    print(f"Approval Probability: {result['approval_probability']:.2%}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    if 'predicted_denial_reasons' in result:
        print(f"Predicted Denial Reasons: {', '.join(result['predicted_denial_reasons'])}")