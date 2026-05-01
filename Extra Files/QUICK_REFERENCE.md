# 🚀 Quick Reference Guide - Financial AI Platform

## 📋 Table of Contents
1. [Installation](#installation)
2. [Running Dashboards](#running-dashboards)
3. [Training Models](#training-models)
4. [Making Predictions](#making-predictions)
5. [Docker Commands](#docker-commands)
6. [Troubleshooting](#troubleshooting)

---

## 🔧 Installation

### Basic Setup
```bash
# Clone and navigate
git clone <repository-url>
cd financial-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Loan Prediction System Setup
```bash
cd loan_prediction_system
python setup.py
cd ..
```

---

## 🎯 Running Dashboards

### Main Executive Dashboard (Port 8501)
```bash
streamlit run src/executive_dashboard.py
# URL: http://localhost:8501
```

### Loan Prediction Dashboard (Port 8502)
```bash
streamlit run loan_prediction_system/src/dashboard.py --server.port 8502
# URL: http://localhost:8502
```

### YoY Analysis Dashboard (Port 8503)
```bash
streamlit run src/enhanced_yoy_dashboard.py --server.port 8503
# URL: http://localhost:8503
```

### Run All Dashboards (Different Terminals)
```bash
# Terminal 1
streamlit run src/executive_dashboard.py

# Terminal 2
streamlit run loan_prediction_system/src/dashboard.py --server.port 8502

# Terminal 3
streamlit run src/enhanced_yoy_dashboard.py --server.port 8503
```

---

## 🤖 Training Models

### Train All Models (Comprehensive Pipeline)
```bash
python -c "
from src.comprehensive_pipeline import LendingPlatformPipeline
pipeline = LendingPlatformPipeline()
results = pipeline.run_full_pipeline()
print('Pipeline completed:', results['status'])
"
```

### Train Loan Prediction Models Only
```bash
cd loan_prediction_system
python src/training_pipeline.py
cd ..
```

### Train Temporal Forecasting Models
```bash
python -c "
from src.hmda_temporal_forecaster import HMDAOpportunityForecaster
forecaster = HMDAOpportunityForecaster()
forecaster.run_pipeline()
"
```

### Configure Model Parameters
```bash
# Edit main config
nano config/config.yaml

# Edit loan prediction config
nano loan_prediction_system/config/model_config.yaml
```

---

## 🔮 Making Predictions

### Loan Outcome Prediction (Python)
```python
from loan_prediction_system.src.prediction_service import get_prediction_service

service = get_prediction_service()

loan_data = {
    'income': 85.0,
    'loan_amount': 350000,
    'property_value': 450000,
    'credit_score': 740,
    'debt_to_income_ratio': 32.0,
    'loan_to_value_ratio': 77.8,
    'applicant_age': 35,
    'loan_purpose': 1,  # 1=Purchase
    'occupancy_type': 1,  # 1=Primary
    'lien_status': 1  # 1=First lien
}

result = service.predict_loan_outcome(loan_data)
print(f"Outcome: {result['outcome']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Risk Score: {result['risk_score']}")
```

### Batch Predictions
```python
import pandas as pd
from loan_prediction_system.src.prediction_service import get_prediction_service

# Load applications
applications = pd.read_csv('loan_applications.csv')

# Predict
service = get_prediction_service()
results = [service.predict_loan_outcome(row.to_dict()) 
           for _, row in applications.iterrows()]

# Save results
pd.DataFrame(results).to_csv('predictions.csv', index=False)
```

### Opportunity Forecasting
```python
from src.opportunity_forecaster import OpportunityForecaster

forecaster = OpportunityForecaster()
forecast = forecaster.forecast_tract_opportunity(
    census_tract='20001952600',
    years_ahead=2
)
print(forecast)
```

---

## 🐳 Docker Commands

### Quick Start
```bash
# Build and run
./docker-run.sh build
./docker-run.sh up

# Or with Docker Compose
docker-compose up -d
```

### All Docker Commands
```bash
./docker-run.sh build      # Build image
./docker-run.sh run        # Run with docker run
./docker-run.sh up         # Start with compose
./docker-run.sh yoy        # Start with YoY service
./docker-run.sh stop       # Stop all services
./docker-run.sh logs       # View logs
./docker-run.sh shell      # Open shell in container
./docker-run.sh clean      # Remove all containers/images
./docker-run.sh rebuild    # Full rebuild
```

### Manual Docker Commands
```bash
# Build
docker build -t financial-ai:latest .

# Run main dashboard
docker run -d \
  --name financial-ai-dashboard \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/config:/app/config \
  financial-ai:latest

# View logs
docker logs -f financial-ai-dashboard

# Stop
docker stop financial-ai-dashboard
docker rm financial-ai-dashboard
```

### Docker Compose
```bash
# Start main service
docker-compose up -d

# Start with YoY analysis
docker-compose --profile yoy up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

---

## 🔍 Troubleshooting

### Check Installation
```bash
# Python version
python --version  # Should be 3.8+

# Streamlit
streamlit --version

# Required packages
pip list | grep -E "pandas|numpy|scikit-learn|xgboost|streamlit"
```

### Verify Data Files
```bash
# Check main data
ls -la data/actual/
# Should have: state_KS.csv, enhanced_census_data.csv, hpi_at_tract.csv

# Check models exist
ls -la data/models/
ls -la loan_prediction_system/models/
```

### View Logs
```bash
# Main pipeline logs
tail -f logs/pipeline.log

# Loan prediction logs
tail -f loan_prediction_system/logs/model_training.log

# Docker logs
docker logs -f financial-ai-dashboard
```

### Port Issues
```bash
# Check what's using port 8501
lsof -i :8501

# Kill process on port
lsof -ti:8501 | xargs kill -9

# Run on different port
streamlit run src/executive_dashboard.py --server.port 8505
```

### Memory Issues
```bash
# Set Python memory
export PYTHONHASHSIZE=0

# Check Docker memory
docker stats

# Increase Docker memory (edit docker-compose.yml)
# deploy:
#   resources:
#     limits:
#       memory: 4G
```

### Model Not Found
```bash
# Train loan prediction models
cd loan_prediction_system
python src/training_pipeline.py

# Verify models created
ls -la models/
# Should have: loan_outcome_model.joblib, denial_reason_model.joblib, preprocessor.joblib
```

### Import Errors
```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src:$(pwd)/loan_prediction_system/src"

# Or use full paths
python -m src.executive_dashboard
```

### Clean Restart
```bash
# Stop everything
pkill -f streamlit
docker stop $(docker ps -q)

# Clean Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Reinstall
pip install --upgrade --force-reinstall -r requirements.txt
```

---

## 📊 Model Performance Check

### Check Loan Prediction Models
```python
from loan_prediction_system.src.model_trainer import ModelTrainer

trainer = ModelTrainer()
trainer.load_models()
metrics = trainer.get_model_metrics()
print(metrics)
```

### Check Forecasting Models
```python
from src.hmda_temporal_forecaster import HMDAOpportunityForecaster

forecaster = HMDAOpportunityForecaster()
performance = forecaster.evaluate_models()
print(performance)
```

---

## 🎯 Common Workflows

### Daily Analysis Workflow
```bash
# 1. Start main dashboard
streamlit run src/executive_dashboard.py

# 2. Start loan prediction (new terminal)
streamlit run loan_prediction_system/src/dashboard.py --server.port 8502

# 3. Access dashboards
# Main: http://localhost:8501
# Loans: http://localhost:8502
```

### Model Retraining Workflow
```bash
# 1. Update data in data/actual/
# 2. Train all models
python -c "from src.comprehensive_pipeline import LendingPlatformPipeline; LendingPlatformPipeline().run_full_pipeline()"

# 3. Train loan models
cd loan_prediction_system
python src/training_pipeline.py
cd ..

# 4. Restart dashboards to use new models
```

### Production Deployment Workflow
```bash
# 1. Build Docker image
./docker-run.sh build

# 2. Test locally
./docker-run.sh up

# 3. Verify at http://localhost:8501

# 4. Deploy with all services
docker-compose --profile yoy up -d

# 5. Monitor
docker-compose logs -f
```

---

## 📱 Dashboard Access URLs

| Dashboard | URL | Port |
|-----------|-----|------|
| Main Executive | http://localhost:8501 | 8501 |
| Loan Prediction | http://localhost:8502 | 8502 |
| YoY Analysis | http://localhost:8503 | 8503 |

---

## 📚 Additional Resources

- [README.md](README.md) - Full documentation
- [DOCKER.md](DOCKER.md) - Docker deployment guide
- [ENHANCED_YOY_ANALYSIS_README.md](ENHANCED_YOY_ANALYSIS_README.md) - YoY analysis details
- [loan_prediction_system/README.md](loan_prediction_system/README.md) - Loan prediction docs

---

**💡 Tip:** Bookmark this page for quick access to all commands!
