#!/usr/bin/env python3
"""
Enhanced HMDA Loan Outcome Prediction System with Advanced Feature Engineering

This module provides a comprehensive loan prediction system specifically designed for HMDA data:
1. Advanced feature engineering for mortgage lending risk assessment
2. Class imbalance handling for realistic approval/denial prediction
3. Fair lending compliance monitoring and bias detection
4. Denial reason prediction for regulatory compliance
5. Comprehensive model validation and interpretability
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

# Local imports
from hmda_feature_engineer import HMDAFeatureEngineer
from hmda_modeling_pipeline import HMDAModelingPipeline
from hmda_denial_reason_predictor import HMDADenialReasonPredictor
from data_validator import DataValidator

# Configuration
# Note: Logging is configured in the main pipeline module
logger = logging.getLogger(__name__)


class EnhancedLoanPredictor:
    """Enhanced HMDA loan outcome prediction with advanced feature engineering and bias handling"""
    
    def __init__(self, config):
        self.config = config
        self.validator = DataValidator()
        
        # Initialize specialized components
        self.feature_engineer = HMDAFeatureEngineer()
        self.modeling_pipeline = HMDAModelingPipeline(random_state=42)
        self.denial_predictor = HMDADenialReasonPredictor(random_state=42)
        
        # Model storage
        self.is_trained = False
        self.feature_columns = []
        self.model_performance = {}
        
        # Training configuration
        self.balance_method = getattr(config, 'balance_method', 'smote')
        self.test_size = getattr(config, 'test_size', 0.2)
        self.cv_folds = getattr(config, 'cv_folds', 5)
        
        logger.info("Enhanced HMDA Loan Predictor initialized with advanced feature engineering")
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare HMDA data with advanced feature engineering"""
        
        logger.info("Loading Kansas HMDA data...")
        
        # Load the actual Kansas HMDA data - DATA_DIR already points to /data/actual
        hmda_path = self.config.DATA_DIR / 'state_KS.csv'
        df = pd.read_csv(hmda_path)
        
        logger.info(f"Loaded {len(df):,} HMDA records")
        
        # Apply comprehensive feature engineering
        df_engineered = self.feature_engineer.engineer_all_features(df)
        
        logger.info(f"Applied feature engineering. New shape: {df_engineered.shape}")
        
        # Log target distribution
        if 'target' in df_engineered.columns:
            target_dist = df_engineered['target'].value_counts()
            logger.info(f"Target distribution: {target_dist.to_dict()}")
        
        return df_engineered
    
    def train_models(self, data_path: str = None) -> Dict:
        """Train enhanced loan prediction models using HMDA data with advanced pipeline"""
        
        logger.info("Training enhanced HMDA loan prediction models...")
        
        # Load and prepare data
        df_engineered = self.load_and_prepare_data()
        
        # Prepare data for modeling
        df_clean, X, y = self.modeling_pipeline.prepare_data(df_engineered)
        
        # Train main loan outcome models
        training_results = self.modeling_pipeline.train_and_evaluate_models(
            X, y, 
            balance_method=self.balance_method,
            test_size=self.test_size,
            cv_folds=self.cv_folds
        )
        
        # Train denial reason models
        logger.info("Training denial reason prediction models...")
        X_denial, y_denial, denied_data = self.denial_predictor.prepare_denial_data(
            df_clean, self.modeling_pipeline.feature_columns
        )
        
        if X_denial is not None:
            denial_results = self.denial_predictor.train_denial_reason_models(X_denial, y_denial)
            logger.info(f"Trained {len(denial_results)} denial reason models")
        else:
            denial_results = {}
            logger.warning("No denial data available for training denial reason models")
        
        # Evaluate fairness
        if self.modeling_pipeline.best_model:
            fairness_results = self.modeling_pipeline.evaluate_fairness(
                df_clean, training_results['X_test'], training_results['y_test']
            )
        else:
            fairness_results = {}
        
        # Store results
        self.model_performance = {
            'loan_outcome_models': training_results['models'],
            'denial_reason_models': denial_results,
            'fairness_analysis': fairness_results,
            'feature_columns': self.modeling_pipeline.feature_columns,
            'data_stats': {
                'total_records': len(df_engineered),
                'training_records': len(df_clean[df_clean['target'].isin([0, 1])]),
                'approval_rate': df_clean[df_clean['target'].isin([0, 1])]['target'].mean(),
                'feature_count': len(self.modeling_pipeline.feature_columns)
            }
        }
        
        self.is_trained = True
        self.feature_columns = self.modeling_pipeline.feature_columns
        
        # Generate performance summary
        summary = self.get_training_summary()
        logger.info(f"Training completed. Best model: {self.get_best_model_name()}")
        
        return summary
    
    def predict_loan_outcome(self, loan_data: Dict) -> Dict:
        """Predict loan outcome with comprehensive analysis"""
        
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_models() first.")
        
        # Main loan outcome prediction
        outcome_result = self.modeling_pipeline.predict_loan_outcome(loan_data)
        
        # If predicted as denied, get denial reasons
        if outcome_result['prediction'] == 'Denied':
            try:
                denial_result = self.denial_predictor.predict_denial_reasons(
                    loan_data, self.feature_columns
                )
            except Exception as e:
                logger.warning(f"Could not predict denial reasons: {str(e)}")
                denial_result = {'predicted_reasons': [], 'explanations': {}}
        else:
            denial_result = {'predicted_reasons': [], 'explanations': {}}
        
        # Combine results
        return {
            **outcome_result,
            'denial_reasons': denial_result['predicted_reasons'],
            'denial_explanations': denial_result['explanations'],
            'primary_denial_reason': denial_result.get('primary_reason'),
            'model_used': self.get_best_model_name()
        }
    
    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary"""
        
        if not self.is_trained:
            return {"status": "Not trained"}
        
        # Model performance summary
        model_summary = self.modeling_pipeline.get_model_summary()
        
        # Denial reason model summary
        denial_summary = self.denial_predictor.get_model_summary()
        
        return {
            'training_status': 'Completed',
            'main_models': model_summary.to_dict('records') if not model_summary.empty else [],
            'denial_models': denial_summary.to_dict('records') if not denial_summary.empty else [],
            'best_model': self.get_best_model_name(),
            'data_statistics': self.model_performance.get('data_stats', {}),
            'feature_count': len(self.feature_columns),
            'fairness_analysis_available': bool(self.model_performance.get('fairness_analysis'))
        }
    
    def get_best_model_name(self) -> str:
        """Get the name of the best performing model"""
        
        if not self.is_trained or not self.model_performance.get('loan_outcome_models'):
            return "None"
        
        models = self.model_performance['loan_outcome_models']
        best_model = max(models.keys(), key=lambda k: models[k]['test_f1'])
        return best_model
    
    def plot_model_performance(self):
        """Generate comprehensive performance visualizations"""
        
        if not self.is_trained:
            logger.warning("No trained models to plot")
            return
        
        # Plot main model performance
        self.modeling_pipeline.plot_model_performance()
        
        # Show denial reason model performance
        denial_summary = self.denial_predictor.get_model_summary()
        if not denial_summary.empty:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            denial_summary.plot(x='Denial_Reason', y='F1_Score', kind='bar', ax=ax)
            ax.set_title('Denial Reason Model Performance')
            ax.set_ylabel('F1 Score')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.show()
    
    def validate_model_logic(self) -> Dict:
        """Validate model logic with extreme test cases"""
        
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_models() first.")
        
        # Define test cases that should definitely be denied
        high_risk_cases = [
            {
                'debt_to_income_ratio': 65.0,
                'loan_to_value_ratio': 0.98,
                'property_value': 200000,
                'income': 30,  # 30k income
                'loan_amount': 190000,
                'loan_purpose': 1,
                'case_description': 'Very high DTI and LTV'
            },
            {
                'debt_to_income_ratio': 55.0,
                'loan_to_value_ratio': 0.95,
                'property_value': 500000,
                'income': 40,  # 40k income  
                'loan_amount': 475000,
                'loan_purpose': 1,
                'case_description': 'High DTI with large loan amount'
            }
        ]
        
        # Define test cases that should likely be approved
        low_risk_cases = [
            {
                'debt_to_income_ratio': 25.0,
                'loan_to_value_ratio': 0.75,
                'property_value': 300000,
                'income': 100,  # 100k income
                'loan_amount': 225000,
                'loan_purpose': 1,
                'case_description': 'Low DTI and LTV with good income'
            },
            {
                'debt_to_income_ratio': 30.0,
                'loan_to_value_ratio': 0.80,
                'property_value': 250000,
                'income': 85,  # 85k income
                'loan_amount': 200000,
                'loan_purpose': 1,
                'case_description': 'Moderate DTI and LTV with stable income'
            }
        ]
        
        validation_results = {
            'high_risk_predictions': [],
            'low_risk_predictions': [],
            'logic_score': 0.0
        }
        
        # Test high risk cases (should be denied)
        correct_denials = 0
        for case in high_risk_cases:
            case_data = {k: v for k, v in case.items() if k != 'case_description'}
            result = self.predict_loan_outcome(case_data)
            
            validation_results['high_risk_predictions'].append({
                'case': case['case_description'],
                'prediction': result['prediction'],
                'probability': result['approval_probability'],
                'correct': result['prediction'] == 'Denied'
            })
            
            if result['prediction'] == 'Denied':
                correct_denials += 1
        
        # Test low risk cases (should be approved)  
        correct_approvals = 0
        for case in low_risk_cases:
            case_data = {k: v for k, v in case.items() if k != 'case_description'}
            result = self.predict_loan_outcome(case_data)
            
            validation_results['low_risk_predictions'].append({
                'case': case['case_description'],
                'prediction': result['prediction'],
                'probability': result['approval_probability'],
                'correct': result['prediction'] == 'Approved'
            })
            
            if result['prediction'] == 'Approved':
                correct_approvals += 1
        
        # Calculate logic score
        total_correct = correct_denials + correct_approvals
        total_cases = len(high_risk_cases) + len(low_risk_cases)
        validation_results['logic_score'] = total_correct / total_cases
        
        logger.info(f"Model logic validation score: {validation_results['logic_score']:.2%}")
        logger.info(f"Correctly denied high-risk cases: {correct_denials}/{len(high_risk_cases)}")
        logger.info(f"Correctly approved low-risk cases: {correct_approvals}/{len(low_risk_cases)}")
        
        return validation_results
    
    def save_models(self, file_path: str):
        """Save trained models to file"""
        
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        model_data = {
            'modeling_pipeline': self.modeling_pipeline,
            'denial_predictor': self.denial_predictor,
            'feature_engineer': self.feature_engineer,
            'model_performance': self.model_performance,
            'feature_columns': self.feature_columns,
            'config': self.config
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Models saved to {file_path}")
    
    def load_models(self, file_path: str):
        """Load trained models from file"""
        
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.modeling_pipeline = model_data['modeling_pipeline']
        self.denial_predictor = model_data['denial_predictor']
        self.feature_engineer = model_data['feature_engineer']
        self.model_performance = model_data['model_performance']
        self.feature_columns = model_data['feature_columns']
        
        self.is_trained = True
        
        logger.info(f"Models loaded from {file_path}")


def train_enhanced_hmda_models(config):
    """Train comprehensive HMDA loan prediction models"""
    
    logger.info("Training enhanced HMDA loan prediction models...")
    
    # Create enhanced predictor
    predictor = EnhancedLoanPredictor(config)
    
    # Train with comprehensive feature engineering
    results = predictor.train_models()
    
    # Save enhanced models
    model_path = config.MODELS_DIR / 'enhanced_hmda_models.pkl'
    predictor.save_models(model_path)
    
    # Validate model logic
    validation_results = predictor.validate_model_logic()
    
    logger.info("Enhanced HMDA model training completed!")
    logger.info(f"Model logic validation score: {validation_results['logic_score']:.2%}")
    
    return predictor, results, validation_results


if __name__ == "__main__":
    # Test the enhanced predictor
    from advanced_lending_platform import LendingConfig
    
    config = LendingConfig()
    predictor, results, validation = train_enhanced_hmda_models(config)
    
    print("\n=== TRAINING RESULTS ===")
    print(f"Best model: {results['best_model']}")
    print(f"Training records: {results['data_statistics']['training_records']:,}")
    print(f"Features used: {results['feature_count']}")
    print(f"Model logic score: {validation['logic_score']:.2%}")
    
    # Test prediction on sample case
    sample_case = {
        'debt_to_income_ratio': 45.0,
        'loan_to_value_ratio': 0.85,
        'property_value': 250000,
        'income': 60,
        'loan_amount': 212500,
        'loan_purpose': 1
    }
    
    prediction = predictor.predict_loan_outcome(sample_case)
    print(f"\n=== SAMPLE PREDICTION ===")
    print(f"Input: DTI=45%, LTV=85%, Income=$60k")
    print(f"Prediction: {prediction['prediction']}")
    print(f"Probability: {prediction['approval_probability']:.3f}")
    print(f"Risk Level: {prediction['risk_level']}")