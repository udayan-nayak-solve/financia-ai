#!/usr/bin/env python3
"""
Quick Start Script for Loan Prediction System
Provides simple commands to run the system
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, description="", check=True):
    """Run a command and optionally check for errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, cwd=Path(__file__).parent)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False


def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("""
🏦 Loan Prediction System - Quick Start

Usage: python run.py [command]

Available commands:
  setup     - Set up the environment and install dependencies
  train     - Train the loan prediction models
  dashboard - Run the prediction dashboard
  test      - Run basic tests
  all       - Run setup, train, and start dashboard

Examples:
  python run.py setup
  python run.py train  
  python run.py dashboard
  python run.py all
        """)
        return
    
    command = sys.argv[1].lower()
    
    if command == "setup":
        print("🏗️ Setting up Loan Prediction System...")
        
        # Install dependencies
        if not run_command("pip install -r requirements.txt", "Installing dependencies"):
            return
        
        # Create necessary directories
        dirs = ["models", "logs", "data"]
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)
        
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. python run.py train    # Train the models")
        print("2. python run.py dashboard # Run the dashboard")
    
    elif command == "train":
        print("🤖 Training loan prediction models...")
        
        # Check if data file exists
        config_path = Path("config/model_config.yaml")
        if not config_path.exists():
            print("❌ Configuration file not found. Please run setup first.")
            return
        
        # Run training
        if run_command("python src/training_pipeline.py", "Training models"):
            print("✅ Model training completed!")
            print("\nModels saved to: ./models/")
            print("Training logs: ./logs/")
            print("\nRun 'python run.py dashboard' to start the prediction interface")
        else:
            print("❌ Training failed. Check logs for details.")
    
    elif command == "dashboard":
        print("🌐 Starting loan prediction dashboard...")
        
        # Check if models exist
        model_dir = Path("models")
        if not model_dir.exists() or not any(model_dir.glob("*.joblib")):
            print("❌ No trained models found. Please run training first:")
            print("   python run.py train")
            return
        
        print("🚀 Starting dashboard on http://localhost:8501")
        print("Press Ctrl+C to stop")
        
        # Run dashboard
        run_command("streamlit run src/dashboard.py", "Starting dashboard", check=False)
    
    elif command == "test":
        print("🧪 Running basic tests...")
        
        # Test imports
        test_script = '''
import sys
sys.path.append("src")

try:
    from config_manager import get_config
    from data_processor import DataProcessor
    from prediction_service import PredictionService
    print("✅ All modules import successfully")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
'''
        
        with open("temp_test.py", "w") as f:
            f.write(test_script)
        
        success = run_command("python temp_test.py", "Testing imports")
        Path("temp_test.py").unlink()
        
        if success:
            print("✅ Basic tests passed!")
        else:
            print("❌ Tests failed!")
    
    elif command == "all":
        print("🚀 Running complete setup and training...")
        
        commands = [
            ("setup", "Setting up environment"),
            ("train", "Training models"),
        ]
        
        for cmd, desc in commands:
            print(f"\n{'='*50}")
            print(f"Running: {desc}")
            print(f"{'='*50}")
            
            # Recursively call this script
            if not run_command(f"python run.py {cmd}", desc):
                print(f"❌ Failed at step: {desc}")
                return
        
        print("\n" + "="*50)
        print("✅ SETUP AND TRAINING COMPLETED!")
        print("="*50)
        print("\nTo start the dashboard:")
        print("python run.py dashboard")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python run.py' for usage instructions")


if __name__ == "__main__":
    main()