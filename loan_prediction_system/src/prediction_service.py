"""
Prediction Service for Loan Outcome Prediction
Handles model loading and prediction logic for the dashboard
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config_manager import get_config
from data_processor import DataProcessor
from model_trainer import ModelManager


class PredictionService:
    """Service for making loan outcome and denial reason predictions"""
    
    def __init__(self, model_dir: str = None):
        """
        Initialize prediction service
        
        Args:
            model_dir: Directory containing saved models
        """
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Set model directory
        if model_dir is None:
            model_dir = self.config.get('persistence.model_dir', './models')
        self.model_dir = Path(model_dir)
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.model_manager = ModelManager()
        self.is_loaded = False
        
        # Denial reason mapping
        self.denial_reasons = {
            1: "Debt-to-income ratio",
            2: "Employment history",
            3: "Credit history", 
            4: "Collateral",
            5: "Insufficient cash (down payment, closing costs)",
            6: "Unverifiable information",
            7: "Credit application incomplete",
            8: "Mortgage insurance denied",
            9: "Other"
        }
        
        # Load models if available
        self._load_models()
    
    def _load_models(self):
        """Load saved models and preprocessor"""
        try:
            # Load preprocessor
            preprocessor_path = self.model_dir / 'preprocessor.joblib'
            if preprocessor_path.exists():
                self.data_processor.load_preprocessor(str(preprocessor_path))
                self.logger.info("Preprocessor loaded successfully")
            else:
                self.logger.warning(f"Preprocessor not found: {preprocessor_path}")
                return
            
            # Load models
            self.model_manager.load_all_models(str(self.model_dir))
            self.is_loaded = True
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            self.is_loaded = False
    
    def validate_input(self, loan_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate input loan data
        
        Args:
            loan_data: Dictionary containing loan application data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = [
            'income', 'loan_amount', 'property_value', 
            'credit_score', 'debt_to_income_ratio', 'loan_to_value_ratio'
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in loan_data:
                return False, f"Missing required field: {field}"
            
            if loan_data[field] is None or loan_data[field] == '':
                return False, f"Field cannot be empty: {field}"
        
        # Validate data ranges
        validation_config = self.config.get('dashboard.validation', {})
        
        validations = [
            ('income', validation_config.get('income', {})),
            ('loan_amount', validation_config.get('loan_amount', {})),
            ('property_value', validation_config.get('property_value', {})),
            ('credit_score', validation_config.get('credit_score', {})),
            ('debt_to_income_ratio', validation_config.get('debt_to_income_ratio', {})),
            ('loan_to_value_ratio', validation_config.get('loan_to_value_ratio', {}))
        ]
        
        for field, limits in validations:
            if field in loan_data:
                value = float(loan_data[field])
                min_val = limits.get('min', float('-inf'))
                max_val = limits.get('max', float('inf'))
                
                if value < min_val or value > max_val:
                    return False, f"{field} must be between {min_val} and {max_val}"
        
        return True, ""
    
    def prepare_input_data(self, loan_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare input data for prediction
        
        Args:
            loan_data: Dictionary containing loan application data
            
        Returns:
            Prepared DataFrame ready for prediction
        """
        # Create DataFrame from input
        df = pd.DataFrame([loan_data])
        
        # Add missing fields with defaults
        defaults = {
            'activity_year': 2025,
            'census_tract': '20015020207',  # Default Kansas census tract
            'action_taken': 1,  # Placeholder, not used for prediction
            'denial_reason': None  # Placeholder, not used for prediction
        }
        
        for field, default_value in defaults.items():
            if field not in df.columns:
                df[field] = default_value
        
        # Transform using data processor
        X = self.data_processor.transform_new_data(df)
        
        return X
    
    def predict_loan_outcome(self, loan_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict loan outcome (approved/denied)
        
        Args:
            loan_data: Dictionary containing loan application data
            
        Returns:
            Dictionary containing prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Please check model directory.")
        
        # Validate input
        is_valid, error_msg = self.validate_input(loan_data)
        if not is_valid:
            raise ValueError(f"Invalid input: {error_msg}")
        
        # Prepare data
        X = self.prepare_input_data(loan_data)
        
        # Make prediction
        predictions, probabilities = self.model_manager.predict_loan_outcome(X)
        
        # Parse results - scikit-learn classes are sorted, so:
        # predictions[0] will be the actual action_taken value (1 or 3)
        # probabilities[0] corresponds to the probability distribution
        prediction = predictions[0]
        
        # Get probabilities - need to map them correctly
        # The classes are sorted: [1, 3] where 1=approved, 3=denied
        if probabilities.shape[1] == 2:
            prob_approved = probabilities[0][0]  # Probability of class 1 (approved)
            prob_denied = probabilities[0][1]    # Probability of class 3 (denied)
        else:
            # Binary case with single probability
            prob_approved = probabilities[0][0] if prediction == 1 else 1 - probabilities[0][0]
            prob_denied = 1 - prob_approved
        
        # Convert prediction to human-readable format
        outcome = "Approved" if prediction == 1 else "Denied"
        
        results = {
            'outcome': outcome,
            'prediction_code': int(prediction),
            'confidence': {
                'approved': float(prob_approved),
                'denied': float(prob_denied)
            },
            'risk_assessment': self._assess_risk_factors(loan_data)
        }
        
        # If denied, predict denial reason
        if prediction == 3:  # Denied
            try:
                denial_reason_code = self.model_manager.predict_denial_reason(X)[0]
                results['denial_reason'] = {
                    'code': int(denial_reason_code),
                    'description': self.denial_reasons.get(denial_reason_code, "Unknown reason")
                }
            except Exception as e:
                self.logger.warning(f"Could not predict denial reason: {e}")
                results['denial_reason'] = {
                    'code': 9,
                    'description': "Other"
                }
        
        return results
    
    def _assess_risk_factors(self, loan_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess individual risk factors
        
        Args:
            loan_data: Loan application data
            
        Returns:
            Risk assessment breakdown
        """
        risk_factors = {}
        
        # Credit score assessment
        credit_score = loan_data.get('credit_score', 0)
        if credit_score >= 740:
            risk_factors['credit_score'] = {'level': 'Low', 'color': 'green'}
        elif credit_score >= 670:
            risk_factors['credit_score'] = {'level': 'Medium', 'color': 'yellow'}
        elif credit_score >= 580:
            risk_factors['credit_score'] = {'level': 'High', 'color': 'orange'}
        else:
            risk_factors['credit_score'] = {'level': 'Very High', 'color': 'red'}
        
        # DTI assessment
        dti = loan_data.get('debt_to_income_ratio', 0)
        if dti <= 28:
            risk_factors['debt_to_income'] = {'level': 'Low', 'color': 'green'}
        elif dti <= 36:
            risk_factors['debt_to_income'] = {'level': 'Medium', 'color': 'yellow'}
        elif dti <= 43:
            risk_factors['debt_to_income'] = {'level': 'High', 'color': 'orange'}
        else:
            risk_factors['debt_to_income'] = {'level': 'Very High', 'color': 'red'}
        
        # LTV assessment
        ltv = loan_data.get('loan_to_value_ratio', 0)
        if ltv <= 80:
            risk_factors['loan_to_value'] = {'level': 'Low', 'color': 'green'}
        elif ltv <= 90:
            risk_factors['loan_to_value'] = {'level': 'Medium', 'color': 'yellow'}
        elif ltv <= 95:
            risk_factors['loan_to_value'] = {'level': 'High', 'color': 'orange'}
        else:
            risk_factors['loan_to_value'] = {'level': 'Very High', 'color': 'red'}
        
        # Income assessment
        income = loan_data.get('income', 0)
        loan_amount = loan_data.get('loan_amount', 0)
        loan_to_income = (loan_amount / (income * 1000)) if income > 0 else float('inf')
        
        if loan_to_income <= 3:
            risk_factors['loan_to_income'] = {'level': 'Low', 'color': 'green'}
        elif loan_to_income <= 4:
            risk_factors['loan_to_income'] = {'level': 'Medium', 'color': 'yellow'}
        elif loan_to_income <= 5:
            risk_factors['loan_to_income'] = {'level': 'High', 'color': 'orange'}
        else:
            risk_factors['loan_to_income'] = {'level': 'Very High', 'color': 'red'}
        
        return risk_factors
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from loan outcome model
        
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_loaded or self.model_manager.loan_outcome_model is None:
            return {}
        
        if hasattr(self.model_manager.loan_outcome_model.model, 'feature_importances_'):
            feature_names = self.data_processor.feature_columns or []
            importances = self.model_manager.loan_outcome_model.model.feature_importances_
            
            if len(feature_names) == len(importances):
                feature_importance = dict(zip(feature_names, importances))
                return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'models_loaded': self.is_loaded,
            'model_directory': str(self.model_dir),
            'available_models': []
        }
        
        if self.is_loaded:
            if self.model_manager.loan_outcome_model:
                info['available_models'].append({
                    'type': 'loan_outcome',
                    'algorithm': getattr(self.model_manager.loan_outcome_model, 'algorithm', 'unknown')
                })
            
            if self.model_manager.denial_reason_model:
                info['available_models'].append({
                    'type': 'denial_reason', 
                    'algorithm': getattr(self.model_manager.denial_reason_model, 'algorithm', 'unknown')
                })
        
        return info
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the prediction service
        
        Returns:
            Health check results
        """
        health = {
            'status': 'healthy',
            'timestamp': pd.Timestamp.now().isoformat(),
            'checks': {}
        }
        
        # Check model loading
        health['checks']['models_loaded'] = {
            'status': 'pass' if self.is_loaded else 'fail',
            'details': 'Models successfully loaded' if self.is_loaded else 'Models not loaded'
        }
        
        # Check model files exist
        model_files = ['loan_outcome_model.joblib', 'preprocessor.joblib']
        for file in model_files:
            file_path = self.model_dir / file
            health['checks'][f'{file}_exists'] = {
                'status': 'pass' if file_path.exists() else 'fail',
                'details': f'File exists: {file_path}' if file_path.exists() else f'File missing: {file_path}'
            }
        
        # Test prediction with sample data
        try:
            sample_data = {
                'income': 75.0,
                'loan_amount': 300000,
                'property_value': 375000,
                'credit_score': 720,
                'debt_to_income_ratio': 28.0,
                'loan_to_value_ratio': 80.0
            }
            
            result = self.predict_loan_outcome(sample_data)
            health['checks']['prediction_test'] = {
                'status': 'pass',
                'details': f'Sample prediction successful: {result["outcome"]}'
            }
        except Exception as e:
            health['checks']['prediction_test'] = {
                'status': 'fail',
                'details': f'Prediction test failed: {str(e)}'
            }
            health['status'] = 'unhealthy'
        
        # Overall status
        failed_checks = [check for check in health['checks'].values() if check['status'] == 'fail']
        if failed_checks:
            health['status'] = 'unhealthy'
        
        return health


# Global prediction service instance
_prediction_service = None


def get_prediction_service(model_dir: str = None) -> PredictionService:
    """
    Get global prediction service instance
    
    Args:
        model_dir: Optional model directory
        
    Returns:
        PredictionService instance
    """
    global _prediction_service
    
    if _prediction_service is None:
        _prediction_service = PredictionService(model_dir)
    
    return _prediction_service