"""
Loan Prediction System - Source Package
Production-grade machine learning system for loan outcome prediction
"""

__version__ = "1.0.0"
__author__ = "Loan Prediction System"
__description__ = "Production-grade ML system for automated loan outcome prediction"

# Import main classes for convenience
try:
    from .config_manager import ConfigManager, get_config
    from .data_processor import DataProcessor
    from .model_trainer import ModelTrainer, ModelManager
    from .prediction_service import PredictionService, get_prediction_service
    from .training_pipeline import TrainingPipeline
    
    __all__ = [
        'ConfigManager', 'get_config',
        'DataProcessor',
        'ModelTrainer', 'ModelManager',
        'PredictionService', 'get_prediction_service',
        'TrainingPipeline'
    ]
    
except ImportError:
    # Handle case where dependencies are not installed
    __all__ = []