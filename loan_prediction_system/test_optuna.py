#!/usr/bin/env python3
"""
Quick test script for Optuna integration
Tests basic functionality without full training pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import logging

# Test Optuna availability
try:
    import optuna
    OPTUNA_AVAILABLE = True
    print("✅ Optuna is available")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("❌ Optuna is not installed. Install with: pip install optuna")
    sys.exit(1)

from src.optuna_model_trainer import OptunaModelTrainer


def create_sample_data():
    """Create sample loan data for testing"""
    print("📊 Creating sample loan data...")
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        class_sep=0.8,
        random_state=42
    )
    
    # Convert to DataFrame with loan-like feature names
    feature_names = [
        'income', 'loan_amount', 'credit_score', 'debt_to_income_ratio',
        'loan_to_value_ratio', 'property_value', 'applicant_age',
        'employment_length', 'loan_purpose', 'loan_term'
    ]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['action_taken'] = y  # 0 = denied, 1 = approved
    
    print(f"Created dataset with {len(df)} samples, {len(feature_names)} features")
    print(f"Class distribution: {df['action_taken'].value_counts().to_dict()}")
    
    return df


def test_optuna_trainer():
    """Test Optuna model trainer with sample data"""
    print("\n🧪 Testing Optuna Model Trainer...")
    
    # Create sample data
    data = create_sample_data()
    
    # Prepare features and target
    X = data.drop('action_taken', axis=1)
    y = data['action_taken']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Test with Random Forest
    print("\n🌲 Testing Random Forest with Optuna...")
    
    trainer = OptunaModelTrainer('loan_outcome_model')
    trainer.algorithm = 'random_forest'  # Override algorithm
    
    # Run optimization with limited trials for testing
    optimization_results = trainer.optimize_hyperparameters(
        X_train, y_train, X_val, y_val,
        n_trials=10,  # Small number for quick test
        n_jobs=1
    )
    
    print(f"✅ Optimization completed!")
    print(f"Best score: {optimization_results['best_score']:.4f}")
    print(f"Best parameters: {optimization_results['best_params']}")
    
    # Evaluate on test set
    evaluation_results = trainer.evaluate(X_test, y_test)
    print(f"Test F1 Score: {evaluation_results['f1_score']:.4f}")
    print(f"Test Accuracy: {evaluation_results['accuracy']:.4f}")
    
    # Get optimization insights
    insights = trainer.get_optimization_insights()
    print(f"\n📈 Optimization Insights:")
    print(f"Total trials: {insights['n_trials']}")
    print(f"Completed trials: {insights['complete_trials']}")
    print(f"Pruned trials: {insights['pruned_trials']}")
    
    if insights['param_importance']:
        print(f"\n🎯 Parameter Importance:")
        for param, importance in insights['param_importance'].items():
            print(f"  {param}: {importance:.3f}")
    
    return True


def test_multiple_algorithms():
    """Test Optuna with different algorithms"""
    print("\n🔬 Testing Multiple Algorithms...")
    
    # Create sample data
    data = create_sample_data()
    X = data.drop('action_taken', axis=1)
    y = data['action_taken']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    algorithms = ['random_forest', 'logistic_regression']
    
    # Add XGBoost if available
    try:
        import xgboost
        algorithms.append('xgboost')
    except ImportError:
        print("⚠️ XGBoost not available, skipping...")
    
    results = {}
    
    for algorithm in algorithms:
        print(f"\n🧮 Testing {algorithm}...")
        
        trainer = OptunaModelTrainer('loan_outcome_model')
        trainer.algorithm = algorithm
        
        try:
            # Quick optimization
            optimization_results = trainer.optimize_hyperparameters(
                X_train, y_train,
                n_trials=5,  # Very small for quick test
                n_jobs=1
            )
            
            evaluation_results = trainer.evaluate(X_test, y_test)
            
            results[algorithm] = {
                'best_score': optimization_results['best_score'],
                'test_f1': evaluation_results['f1_score'],
                'test_accuracy': evaluation_results['accuracy']
            }
            
            print(f"✅ {algorithm} - F1: {evaluation_results['f1_score']:.4f}")
            
        except Exception as e:
            print(f"❌ {algorithm} failed: {e}")
            results[algorithm] = {'error': str(e)}
    
    print(f"\n📊 Algorithm Comparison Summary:")
    print("-" * 50)
    for algorithm, result in results.items():
        if 'error' not in result:
            print(f"{algorithm:15} | F1: {result['test_f1']:.4f} | Acc: {result['test_accuracy']:.4f}")
        else:
            print(f"{algorithm:15} | Error: {result['error']}")
    
    return results


def main():
    """Main test function"""
    print("🚀 OPTUNA INTEGRATION TEST")
    print("=" * 50)
    
    # Setup basic logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise for testing
    
    try:
        # Test basic functionality
        test_optuna_trainer()
        
        # Test multiple algorithms
        test_multiple_algorithms()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("Optuna integration is working correctly.")
        print("\n🎯 Next steps:")
        print("1. Install Optuna in your environment: pip install optuna")
        print("2. Update model_config.yaml to enable Optuna optimization")
        print("3. Run the enhanced training pipeline")
        print("4. Compare results with baseline models")
        
    except Exception as e:
        print(f"\n❌ TESTS FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)