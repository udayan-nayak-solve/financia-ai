#!/usr/bin/env python3
"""
Setup and Installation Script for Loan Prediction System
Handles environment setup, dependency installation, and initial configuration
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description=""):
    """Run a shell command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description or command}")
    print(f"{'='*50}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {description or command}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    else:
        print(f"✅ Success: {description or command}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True


def setup_environment():
    """Set up Python virtual environment"""
    print("🔧 Setting up Python virtual environment...")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("Virtual environment already exists.")
        return True
    
    # Create virtual environment
    if not run_command("python3 -m venv venv", "Creating virtual environment"):
        return False
    
    # Activate and upgrade pip
    if sys.platform == "win32":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    return True


def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    if sys.platform == "win32":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    # Install core dependencies
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        return False
    
    return True


def setup_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "models/checkpoints",
        "logs",
        "tests/unit",
        "tests/integration",
        "config/environments"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    return True


def create_env_file():
    """Create environment configuration file"""
    print("🔧 Creating environment configuration...")
    
    env_content = """# Loan Prediction System Environment Configuration
# Copy this file to .env and customize as needed

# Model Configuration
MODEL_DIR=./models
DATA_DIR=./data
LOG_LEVEL=INFO

# Dashboard Configuration  
DASHBOARD_PORT=8501
DASHBOARD_HOST=localhost

# API Configuration (if using FastAPI)
API_PORT=8000
API_HOST=localhost

# Monitoring and Logging
ENABLE_MONITORING=true
LOG_TO_FILE=true

# Development Settings
DEBUG=false
TESTING=false
"""
    
    env_file = Path(".env.example")
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"  ✅ Created: {env_file}")
    print("  ℹ️  Copy .env.example to .env and customize as needed")
    
    return True


def run_tests():
    """Run basic tests to verify installation"""
    print("🧪 Running basic tests...")
    
    if sys.platform == "win32":
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "venv/bin/python"
    
    # Test imports
    test_script = """
import sys
print("Testing core imports...")

try:
    import pandas as pd
    import numpy as np
    import sklearn
    import yaml
    import streamlit
    import plotly
    print("✅ All core dependencies imported successfully")
    
    # Test configuration loading
    sys.path.append('src')
    from config_manager import get_config
    config = get_config()
    print("✅ Configuration manager working")
    
    print("✅ All tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
"""
    
    # Write test script to temporary file
    test_file = Path("test_installation.py")
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    # Run test
    success = run_command(f"{python_cmd} test_installation.py", "Testing installation")
    
    # Clean up
    test_file.unlink()
    
    return success


def create_run_scripts():
    """Create convenience scripts for running the system"""
    print("📝 Creating run scripts...")
    
    # Training script
    train_script = """#!/bin/bash
# Training Script for Loan Prediction System

echo "🚀 Starting Loan Prediction Model Training..."

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Run training pipeline
python src/training_pipeline.py

echo "✅ Training completed!"
"""
    
    with open("train_model.sh", 'w') as f:
        f.write(train_script)
    
    # Dashboard script
    dashboard_script = """#!/bin/bash
# Dashboard Script for Loan Prediction System

echo "🚀 Starting Loan Prediction Dashboard..."

# Activate virtual environment
if [[ "$OSTYPE" == "msys" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Run dashboard
streamlit run src/dashboard.py

echo "✅ Dashboard started!"
"""
    
    with open("run_dashboard.sh", 'w') as f:
        f.write(dashboard_script)
    
    # Make scripts executable on Unix systems
    if sys.platform != "win32":
        os.chmod("train_model.sh", 0o755)
        os.chmod("run_dashboard.sh", 0o755)
    
    print("  ✅ Created: train_model.sh")
    print("  ✅ Created: run_dashboard.sh")
    
    return True


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup Loan Prediction System")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-venv", action="store_true", help="Skip virtual environment setup")
    args = parser.parse_args()
    
    print("🏦 Loan Prediction System Setup")
    print("="*50)
    
    # Setup steps
    setup_steps = [
        ("Virtual Environment", setup_environment, not args.skip_venv),
        ("Dependencies", install_dependencies, True),
        ("Directories", setup_directories, True),
        ("Environment File", create_env_file, True),
        ("Run Scripts", create_run_scripts, True),
        ("Tests", run_tests, not args.skip_tests)
    ]
    
    failed_steps = []
    
    for step_name, step_function, should_run in setup_steps:
        if should_run:
            print(f"\n{'='*20} {step_name} {'='*20}")
            if not step_function():
                failed_steps.append(step_name)
                print(f"❌ Failed: {step_name}")
            else:
                print(f"✅ Completed: {step_name}")
        else:
            print(f"⏭️  Skipped: {step_name}")
    
    # Summary
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    
    if failed_steps:
        print(f"❌ Setup completed with {len(failed_steps)} failed steps:")
        for step in failed_steps:
            print(f"   • {step}")
        print("\nPlease review the errors above and re-run setup.")
        return 1
    else:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and customize")
        print("2. Update config/model_config.yaml with your data path")
        print("3. Run: ./train_model.sh (or python src/training_pipeline.py)")
        print("4. Run: ./run_dashboard.sh (or streamlit run src/dashboard.py)")
        return 0


if __name__ == "__main__":
    sys.exit(main())