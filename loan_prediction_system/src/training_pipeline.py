"""
Main Training Pipeline for Loan Prediction System
Orchestrates the complete model training workflow
"""

import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config_manager import get_config, setup_logging
from data_processor import DataProcessor
from model_trainer import ModelManager


class TrainingPipeline:
    """Main training pipeline orchestrator"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize training pipeline
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config()
        self.logger = setup_logging(self.config)
        self.logger.info("Initializing Loan Prediction Training Pipeline")
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.model_manager = ModelManager()
        
        # Results storage
        self.results = {}
        
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Returns:
            Dictionary containing all results
        """
        try:
            self.logger.info("="*50)
            self.logger.info("STARTING LOAN PREDICTION MODEL TRAINING PIPELINE")
            self.logger.info("="*50)
            
            # Step 1: Load and validate data
            self.logger.info("Step 1: Loading and validating data")
            df = self.data_processor.load_data()
            df_clean = self.data_processor.validate_data(df)
            
            self.results['data_info'] = {
                'original_shape': df.shape,
                'cleaned_shape': df_clean.shape,
                'rows_removed': df.shape[0] - df_clean.shape[0],
                'columns': df_clean.columns.tolist()
            }
            
            # Step 2: Feature engineering
            self.logger.info("Step 2: Creating features")
            df_features = self.data_processor.create_features(df_clean)
            
            # Step 3: Prepare features for modeling
            self.logger.info("Step 3: Preparing features for modeling")
            X, y = self.data_processor.prepare_features(df_features, 'action_taken', is_training=True)
            
            self.results['feature_info'] = {
                'num_features': len(X.columns),
                'feature_names': X.columns.tolist(),
                'target_distribution': y.value_counts().to_dict()
            }
            
            # Step 4: Split data
            self.logger.info("Step 4: Splitting data into train/test sets")
            X_train, X_test, y_train, y_test = self.data_processor.split_data(X, y)
            
            # Step 5: Scale features
            self.logger.info("Step 5: Scaling features")
            X_train_scaled, X_test_scaled = self.data_processor.scale_features(X_train, X_test)
            
            # Step 6: Train models
            self.logger.info("Step 6: Training models")
            training_results = self.model_manager.train_all_models(
                self.data_processor, X_train_scaled, y_train, X_test_scaled, y_test
            )
            
            self.results['model_results'] = training_results
            
            # Step 7: Save models and preprocessor
            self.logger.info("Step 7: Saving models and preprocessor")
            model_dir = Path(self.config.get('persistence.model_dir', './models'))
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save models
            saved_paths = self.model_manager.save_all_models(str(model_dir))
            
            # Save preprocessor
            preprocessor_path = model_dir / 'preprocessor.joblib'
            self.data_processor.save_preprocessor(str(preprocessor_path))
            saved_paths['preprocessor'] = str(preprocessor_path)
            
            self.results['saved_models'] = saved_paths
            
            # Step 8: Generate training report
            self.logger.info("Step 8: Generating training report")
            report_path = self._generate_training_report()
            self.results['report_path'] = report_path
            
            self.logger.info("="*50)
            self.logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("="*50)
            
            # Print summary
            self._print_summary()
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
            raise
    
    def _generate_training_report(self) -> str:
        """
        Generate comprehensive training report
        
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"./logs/training_report_{timestamp}.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare report data
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_config': {
                'data_file': self.config.get('data.input_file'),
                'test_size': self.config.get('data.test_size'),
                'random_state': self.config.get('data.random_state')
            },
            'data_summary': self.results.get('data_info', {}),
            'feature_summary': self.results.get('feature_info', {}),
            'model_results': self.results.get('model_results', {}),
            'saved_models': self.results.get('saved_models', {})
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Training report saved to: {report_path}")
        return str(report_path)
    
    def _print_summary(self):
        """Print training summary to console"""
        print("\n" + "="*60)
        print("TRAINING PIPELINE SUMMARY")
        print("="*60)
        
        # Data summary
        data_info = self.results.get('data_info', {})
        print(f"📊 Data Summary:")
        print(f"   Original records: {data_info.get('original_shape', ['N/A'])[0]:,}")
        print(f"   Clean records: {data_info.get('cleaned_shape', ['N/A'])[0]:,}")
        print(f"   Features created: {self.results.get('feature_info', {}).get('num_features', 'N/A')}")
        
        # Model performance
        model_results = self.results.get('model_results', {})
        
        if 'loan_outcome' in model_results:
            loan_eval = model_results['loan_outcome'].get('evaluation', {})
            print(f"\n🎯 Loan Outcome Model Performance:")
            print(f"   Accuracy: {loan_eval.get('accuracy', 0):.4f}")
            print(f"   F1 Score: {loan_eval.get('f1_score', 0):.4f}")
            print(f"   ROC AUC: {loan_eval.get('roc_auc', 'N/A')}")
        
        if 'denial_reason' in model_results:
            denial_eval = model_results['denial_reason'].get('evaluation', {})
            print(f"\n❌ Denial Reason Model Performance:")
            print(f"   Accuracy: {denial_eval.get('accuracy', 0):.4f}")
            print(f"   F1 Score: {denial_eval.get('f1_score', 0):.4f}")
        
        # Saved models
        saved_models = self.results.get('saved_models', {})
        print(f"\n💾 Saved Models:")
        for model_type, path in saved_models.items():
            print(f"   {model_type}: {path}")
        
        print("\n✅ Pipeline completed successfully!")
        print("="*60)


def main():
    """Main entry point"""
    try:
        pipeline = TrainingPipeline()
        results = pipeline.run_pipeline()
        return results
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()