"""
Configuration Manager for Loan Prediction System
Handles loading and validation of configuration files
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging


PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def resolve_package_path(path: Union[str, Path]) -> Path:
    """Resolve a path against the package root unless it's already absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (PACKAGE_ROOT / p).resolve()


class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def _validate_config(self):
        """Validate configuration structure and values"""
        required_sections = ['data', 'features', 'loan_outcome_model', 'logging']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate data configuration
        data_config = self.config['data']
        if 'test_size' not in data_config or not 0 < data_config['test_size'] < 1:
            raise ValueError("Invalid test_size in data configuration")
        
        # Validate model configuration
        loan_model_config = self.config['loan_outcome_model']
        if 'algorithm' not in loan_model_config:
            raise ValueError("Algorithm not specified in loan_outcome_model")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (supports dot notation like 'data.test_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def update(self, key: str, value: Any):
        """
        Update configuration value using dot notation
        
        Args:
            key: Configuration key
            value: New value
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """
        Save configuration to file
        
        Args:
            path: Optional custom path to save configuration
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
    
    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """
        Get model parameters for specified algorithm
        
        Args:
            model_type: Type of model ('loan_outcome_model' or 'denial_reason_model')
            
        Returns:
            Model parameters dictionary
        """
        model_config = self.config.get(model_type, {})
        algorithm = model_config.get('algorithm', 'random_forest')
        
        return model_config.get(algorithm, {})
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration"""
        return self.config.get('features', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config.get('training', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration"""
        return self.config.get('evaluation', {})


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """Get global configuration manager instance"""
    return config_manager


def setup_logging(config: ConfigManager):
    """
    Setup logging based on configuration
    
    Args:
        config: Configuration manager instance
    """
    log_config = config.get('logging', {})

    log_file = resolve_package_path(log_config.get('file', './logs/model_training.log'))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler() if log_config.get('console', True) else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)