"""
Enhanced Training Pipeline with Optuna Integration
Provides both standard and advanced hyperparameter optimization
"""

import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from data_processor import DataProcessor
from model_trainer import ModelTrainer
from optuna_model_trainer import OptunaModelTrainer
from config_manager import get_config


class EnhancedTrainingPipeline:
    """Enhanced training pipeline with Optuna optimization support"""
    
    def __init__(self):
        """Initialize the enhanced training pipeline"""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.data_processor = DataProcessor()
        self.results = {}
        
        # Create directories
        Path("models").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
    
    def run_pipeline(self, use_optuna: Optional[bool] = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Args:
            use_optuna: Whether to use Optuna optimization (None = use config setting)
            
        Returns:
            Complete training results
        """
        start_time = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info("ENHANCED LOAN PREDICTION MODEL TRAINING PIPELINE")
        self.logger.info("=" * 60)
        
        try:
            # Step 1: Load and prepare data
            self.logger.info("Step 1: Loading and preparing data...")
            data = self._load_and_prepare_data()
            
            # Step 2: Check optimization method
            training_config = self.config.get_training_config()
            tuning_config = training_config.get('hyperparameter_tuning', {})
            
            if use_optuna is None:
                use_optuna = (tuning_config.get('enabled', False) and 
                            tuning_config.get('method', 'grid_search') == 'optuna')
            
            # Step 3: Train models
            if use_optuna:
                self.logger.info("Step 2: Training models with Optuna optimization...")
                model_results = self._train_models_with_optuna(data)
            else:
                self.logger.info("Step 2: Training models with standard methods...")
                model_results = self._train_models_standard(data)
            
            # Step 4: Save results
            self.logger.info("Step 3: Saving models and results...")
            self._save_results(model_results)
            
            # Step 5: Generate summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            summary = self._generate_summary(model_results, duration, use_optuna)
            
            self.logger.info("=" * 60)
            self.logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total Duration: {duration:.2f} seconds")
            self.logger.info("=" * 60)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _load_and_prepare_data(self) -> Dict[str, Any]:
        """Load and prepare training data"""
        # Load data
        data = self.data_processor.load_data()
        self.logger.info(f"Loaded {len(data)} records")
        
        # Validate data
        data = self.data_processor.validate_data(data)
        self.logger.info(f"After validation: {len(data)} records")
        
        # Create features
        data = self.data_processor.create_features(data)
        self.logger.info(f"Created {len(data.columns)} features")
        
        # Prepare features for loan outcome prediction
        X, y = self.data_processor.prepare_features(data, 'action_taken', is_training=True)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.get('data.test_size', 0.2),
            random_state=self.config.get('data.random_state', 42),
            stratify=y
        )
        
        # Further split training into train/validation for Optuna
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config.get('data.validation_size', 0.1),
            random_state=self.config.get('data.random_state', 42),
            stratify=y_train
        )
        
        # Scale features
        X_train, X_test = self.data_processor.scale_features(X_train, X_test)
        X_train, X_val = self.data_processor.scale_features(X_train, X_val)
        
        self.logger.info(f"Training set: {len(X_train)} samples")
        self.logger.info(f"Validation set: {len(X_val)} samples") 
        self.logger.info(f"Test set: {len(X_test)} samples")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': X_train.columns.tolist(),
            'original_data': data
        }
    
    def _train_models_with_optuna(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train models using Optuna optimization"""
        results = {}
        
        # Get Optuna configuration
        training_config = self.config.get_training_config()
        optuna_config = training_config.get('hyperparameter_tuning', {}).get('optuna', {})
        
        n_trials = optuna_config.get('n_trials', 100)
        timeout = optuna_config.get('timeout', 3600)  # 1 hour
        n_jobs = optuna_config.get('n_jobs', 1)
        
        # Train loan outcome model
        self.logger.info("Training loan outcome model with Optuna...")
        loan_trainer = OptunaModelTrainer('loan_outcome_model')
        
        # Optimize hyperparameters
        optimization_results = loan_trainer.optimize_hyperparameters(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs
        )
        
        # Evaluate on test set
        evaluation_results = loan_trainer.evaluate(data['X_test'], data['y_test'])\n        
        # Get feature importance
        feature_importance = loan_trainer.get_feature_importance(data['feature_names'])
        
        # Get optimization insights
        optimization_insights = loan_trainer.get_optimization_insights()
        
        results['loan_outcome'] = {
            'optimization': optimization_results,
            'evaluation': evaluation_results,
            'feature_importance': feature_importance,
            'insights': optimization_insights,
            'trainer': loan_trainer
        }
        
        # Save optimization plots
        try:
            plots_dir = Path("logs/optimization_plots")
            plots_dir.mkdir(exist_ok=True)
            
            loan_trainer.plot_optimization_history(
                str(plots_dir / "loan_outcome_optimization_history.png")
            )
            loan_trainer.plot_param_importances(
                str(plots_dir / "loan_outcome_param_importance.png")
            )
        except Exception as e:
            self.logger.warning(f"Could not save optimization plots: {e}")
        
        # Train denial reason model if possible
        denied_mask = data['y_train'] == 3
        if denied_mask.sum() > 50:
            self.logger.info("Training denial reason model with Optuna...")
            
            try:
                # Prepare denial reason data
                denial_data = self._prepare_denial_reason_data(data['original_data'])
                
                if denial_data is not None:
                    denial_trainer = OptunaModelTrainer('denial_reason_model')
                    
                    # Use smaller number of trials for denial reason model
                    denial_trials = min(n_trials // 2, 50)
                    
                    denial_optimization = denial_trainer.optimize_hyperparameters(
                        denial_data['X_train'], denial_data['y_train'],
                        denial_data['X_val'], denial_data['y_val'],
                        n_trials=denial_trials,
                        timeout=timeout // 2,
                        n_jobs=n_jobs
                    )
                    
                    denial_evaluation = denial_trainer.evaluate(
                        denial_data['X_test'], denial_data['y_test']
                    )
                    
                    denial_importance = denial_trainer.get_feature_importance(
                        denial_data['feature_names']
                    )
                    
                    results['denial_reason'] = {
                        'optimization': denial_optimization,
                        'evaluation': denial_evaluation,
                        'feature_importance': denial_importance,
                        'insights': denial_trainer.get_optimization_insights(),
                        'trainer': denial_trainer
                    }
                    
            except Exception as e:
                self.logger.warning(f"Failed to train denial reason model: {e}")
        
        return results
    
    def _train_models_standard(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train models using standard methods"""
        results = {}
        
        # Train loan outcome model
        self.logger.info("Training loan outcome model with standard methods...")
        loan_trainer = ModelTrainer('loan_outcome_model')
        
        # Check if hyperparameter tuning is enabled
        training_config = self.config.get_training_config()
        tuning_config = training_config.get('hyperparameter_tuning', {})
        
        if tuning_config.get('enabled', False):
            # Use built-in hyperparameter tuning
            tuning_results = loan_trainer.hyperparameter_tuning(
                data['X_train'], data['y_train']
            )
        else:
            # Train with default parameters
            training_results = loan_trainer.train(
                data['X_train'], data['y_train'],
                data['X_val'], data['y_val']
            )
            tuning_results = {'method': 'default_params'}
        
        # Evaluate model
        evaluation_results = loan_trainer.evaluate(data['X_test'], data['y_test'])
        feature_importance = loan_trainer.get_feature_importance(data['feature_names'])
        
        results['loan_outcome'] = {
            'training': tuning_results,
            'evaluation': evaluation_results,
            'feature_importance': feature_importance,
            'trainer': loan_trainer
        }
        
        return results
    
    def _prepare_denial_reason_data(self, original_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Prepare data for denial reason model training"""
        try:
            # Filter for denied loans with valid denial reasons
            denied_data = original_data[original_data['action_taken'] == 3].copy()
            denied_data = denied_data.dropna(subset=['denial_reason'])
            
            if len(denied_data) < 50:
                self.logger.warning("Not enough denied loans for denial reason model")
                return None
            
            # Prepare features
            X, y = self.data_processor.prepare_features(
                denied_data, 'denial_reason', is_training=True
            )
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
            )
            
            # Scale features
            X_train, X_test = self.data_processor.scale_features(X_train, X_test)
            X_train, X_val = self.data_processor.scale_features(X_train, X_val)
            
            return {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'feature_names': X_train.columns.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to prepare denial reason data: {e}")
            return None
    
    def _save_results(self, model_results: Dict[str, Any]):
        """Save trained models and results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        for model_type, results in model_results.items():
            if 'trainer' in results:
                trainer = results['trainer']
                model_path = trainer.save_model("models")
                self.logger.info(f"Saved {model_type} model to: {model_path}")
        
        # Save training report
        report = {
            'timestamp': timestamp,
            'config': dict(self.config.config),
            'results': {
                model_type: {
                    k: v for k, v in results.items() 
                    if k != 'trainer'  # Don't serialize trainer objects
                }
                for model_type, results in model_results.items()
            }
        }
        
        report_path = f"logs/training_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Training report saved to: {report_path}")
    
    def _generate_summary(
        self, 
        model_results: Dict[str, Any], 
        duration: float,
        used_optuna: bool
    ) -> Dict[str, Any]:
        """Generate training summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'optimization_method': 'optuna' if used_optuna else 'standard',
            'models_trained': list(model_results.keys()),
            'performance': {}
        }
        
        # Extract performance metrics
        for model_type, results in model_results.items():
            if 'evaluation' in results:
                eval_results = results['evaluation']
                summary['performance'][model_type] = {
                    'accuracy': eval_results.get('accuracy', 0.0),
                    'f1_score': eval_results.get('f1_score', 0.0),
                    'precision': eval_results.get('precision', 0.0),
                    'recall': eval_results.get('recall', 0.0)
                }
                
                if 'roc_auc' in eval_results:
                    summary['performance'][model_type]['roc_auc'] = eval_results['roc_auc']
        
        # Add optimization insights if Optuna was used
        if used_optuna:
            summary['optimization_insights'] = {}
            for model_type, results in model_results.items():
                if 'insights' in results:
                    insights = results['insights']
                    summary['optimization_insights'][model_type] = {
                        'best_score': insights.get('best_score', 0.0),
                        'n_trials': insights.get('n_trials', 0),
                        'pruned_trials': insights.get('pruned_trials', 0),
                        'param_importance': insights.get('param_importance', {})
                    }
        
        return summary
    
    def compare_optimization_methods(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Compare standard vs Optuna optimization methods
        
        Args:
            n_trials: Number of trials for Optuna
            
        Returns:
            Comparison results
        """
        self.logger.info("Comparing optimization methods...")
        
        # Load and prepare data once
        data = self._load_and_prepare_data()
        
        # Train with standard methods
        self.logger.info("Training with standard methods...")
        standard_results = self._train_models_standard(data)
        
        # Train with Optuna (limited trials for comparison)
        self.logger.info(f"Training with Optuna ({n_trials} trials)...")
        
        # Temporarily update config for limited trials
        original_config = self.config.get_training_config()
        self.config.config['training']['hyperparameter_tuning']['optuna']['n_trials'] = n_trials
        
        optuna_results = self._train_models_with_optuna(data)
        
        # Restore original config
        self.config.config['training'] = original_config
        
        # Compare results
        comparison = {
            'standard': {
                'loan_outcome': {
                    'accuracy': standard_results['loan_outcome']['evaluation']['accuracy'],
                    'f1_score': standard_results['loan_outcome']['evaluation']['f1_score']
                }
            },
            'optuna': {
                'loan_outcome': {
                    'accuracy': optuna_results['loan_outcome']['evaluation']['accuracy'],
                    'f1_score': optuna_results['loan_outcome']['evaluation']['f1_score'],
                    'best_score': optuna_results['loan_outcome']['optimization']['best_score'],
                    'n_trials': optuna_results['loan_outcome']['optimization']['n_trials']
                }
            }
        }
        
        # Calculate improvement
        standard_f1 = standard_results['loan_outcome']['evaluation']['f1_score']
        optuna_f1 = optuna_results['loan_outcome']['evaluation']['f1_score']
        improvement = ((optuna_f1 - standard_f1) / standard_f1) * 100
        
        comparison['improvement'] = {
            'f1_score_improvement_percent': improvement,
            'absolute_improvement': optuna_f1 - standard_f1
        }
        
        self.logger.info(f"Optuna F1 improvement: {improvement:.2f}%")
        
        return comparison


def main():
    """Main function to run the enhanced training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Loan Prediction Model Training')
    parser.add_argument('--optuna', action='store_true', help='Use Optuna optimization')
    parser.add_argument('--compare', action='store_true', help='Compare optimization methods')
    parser.add_argument('--trials', type=int, default=100, help='Number of Optuna trials')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/enhanced_training.log'),
            logging.StreamHandler()
        ]
    )
    
    pipeline = EnhancedTrainingPipeline()
    
    if args.compare:
        results = pipeline.compare_optimization_methods(n_trials=args.trials)
        print("\n" + "="*60)
        print("OPTIMIZATION METHOD COMPARISON")
        print("="*60)
        print(f"Standard F1 Score: {results['standard']['loan_outcome']['f1_score']:.4f}")
        print(f"Optuna F1 Score: {results['optuna']['loan_outcome']['f1_score']:.4f}")
        print(f"Improvement: {results['improvement']['f1_score_improvement_percent']:.2f}%")
        print("="*60)
    else:
        results = pipeline.run_pipeline(use_optuna=args.optuna)
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"Method: {results['optimization_method']}")
        print(f"Duration: {results['duration_seconds']:.2f} seconds")
        
        for model_type, perf in results['performance'].items():
            print(f"\n{model_type.upper()} MODEL:")
            print(f"  Accuracy: {perf['accuracy']:.4f}")
            print(f"  F1 Score: {perf['f1_score']:.4f}")


if __name__ == '__main__':
    main()