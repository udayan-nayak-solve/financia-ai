#!/usr/bin/env python3
"""
Setup script for Optuna integration
Installs dependencies and sets up the enhanced training pipeline
"""

import subprocess
import sys
import os
from pathlib import Path


def install_optuna():
    """Install Optuna and related dependencies"""
    print("📦 Installing Optuna and dependencies...")
    
    packages = [
        'optuna>=3.0.0',
        'matplotlib>=3.5.0',  # For optimization plots
        'plotly>=5.10.0',     # For interactive plots
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True


def update_config():
    """Update model configuration to enable Optuna"""
    config_path = Path('config/model_config.yaml')
    
    if not config_path.exists():
        print(f"⚠️ Configuration file not found: {config_path}")
        return False
    
    # Read current config
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Check if Optuna is already enabled
    if 'method: "optuna"' in content:
        print("✅ Optuna is already enabled in configuration")
        return True
    
    # Update configuration to enable Optuna
    updated_content = content.replace(
        'enabled: false',
        'enabled: true'
    ).replace(
        'method: "grid_search"',
        'method: "optuna"'
    )
    
    # Backup original config
    backup_path = config_path.with_suffix('.yaml.backup')
    with open(backup_path, 'w') as f:
        f.write(content)
    
    # Write updated config
    with open(config_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✅ Updated configuration file (backup saved to {backup_path})")
    return True


def test_installation():
    """Test Optuna installation"""
    print("🧪 Testing Optuna installation...")
    
    try:
        import optuna
        print(f"✅ Optuna {optuna.__version__} is installed and working")
        
        # Quick test study
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            return (x - 2) ** 2
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10, show_progress_bar=False)
        
        print(f"✅ Optuna optimization test passed (best value: {study.best_value:.4f})")
        return True
        
    except ImportError:
        print("❌ Optuna import failed")
        return False
    except Exception as e:
        print(f"❌ Optuna test failed: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    directories = [
        'logs/optimization_plots',
        'models/optimized',
        'logs/studies'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def generate_example_script():
    """Generate example usage script"""
    example_script = '''#!/usr/bin/env python3
"""
Example script for using Optuna optimization
Run this after setup to test the enhanced training pipeline
"""

import sys
import os
sys.path.append('src')

from enhanced_training_pipeline import EnhancedTrainingPipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def main():
    """Run Optuna optimization example"""
    print("🚀 Running Optuna Optimization Example")
    print("=" * 50)
    
    # Initialize enhanced pipeline
    pipeline = EnhancedTrainingPipeline()
    
    # Run with Optuna optimization (limited trials for demo)
    results = pipeline.run_pipeline(use_optuna=True)
    
    print("\\n" + "=" * 50)
    print("📊 RESULTS SUMMARY")
    print("=" * 50)
    
    print(f"Optimization Method: {results['optimization_method']}")
    print(f"Training Duration: {results['duration_seconds']:.2f} seconds")
    
    for model_type, performance in results['performance'].items():
        print(f"\\n{model_type.upper()} MODEL:")
        print(f"  Accuracy: {performance['accuracy']:.4f}")
        print(f"  F1 Score: {performance['f1_score']:.4f}")
        print(f"  Precision: {performance['precision']:.4f}")
        print(f"  Recall: {performance['recall']:.4f}")
    
    # Show optimization insights
    if 'optimization_insights' in results:
        print("\\n🎯 OPTIMIZATION INSIGHTS:")
        for model_type, insights in results['optimization_insights'].items():
            print(f"\\n{model_type.upper()}:")
            print(f"  Best Score: {insights['best_score']:.4f}")
            print(f"  Total Trials: {insights['n_trials']}")
            print(f"  Pruned Trials: {insights['pruned_trials']}")
            
            if insights['param_importance']:
                print("  Top Parameters:")
                for param, importance in list(insights['param_importance'].items())[:3]:
                    print(f"    {param}: {importance:.3f}")

if __name__ == '__main__':
    main()
'''
    
    with open('run_optuna_example.py', 'w') as f:
        f.write(example_script)
    
    # Make executable
    os.chmod('run_optuna_example.py', 0o755)
    print("✅ Created example script: run_optuna_example.py")


def main():
    """Main setup function"""
    print("🚀 OPTUNA INTEGRATION SETUP")
    print("=" * 50)
    print("Setting up enhanced hyperparameter optimization with Optuna...")
    print()
    
    # Check if we're in the right directory
    if not Path('src').exists() or not Path('config').exists():
        print("❌ Please run this script from the loan_prediction_system directory")
        return False
    
    success = True
    
    # Step 1: Install Optuna
    if not install_optuna():
        success = False
    
    # Step 2: Test installation
    if success and not test_installation():
        success = False
    
    # Step 3: Create directories
    if success:
        create_directories()
    
    # Step 4: Update configuration
    if success and not update_config():
        success = False
    
    # Step 5: Generate example script
    if success:
        generate_example_script()
    
    print("\\n" + "=" * 50)
    if success:
        print("✅ SETUP COMPLETED SUCCESSFULLY!")
        print()
        print("🎯 Next Steps:")
        print("1. Run the test: python test_optuna.py")
        print("2. Try the example: python run_optuna_example.py")
        print("3. Compare methods: python compare_optimization.py --compare")
        print("4. Use in your training: python src/enhanced_training_pipeline.py --optuna")
        print()
        print("📚 Benefits of Optuna:")
        print("- Faster convergence to optimal parameters")
        print("- Automatic pruning of poor trials")
        print("- Better parameter space exploration")
        print("- Visualization and analysis tools")
        print("- Expected 1-5% improvement in model performance")
    else:
        print("❌ SETUP FAILED!")
        print("Please check the error messages above and try again.")
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)