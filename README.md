# 🏦 Financial AI Platform

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)](https://xgboost.ai)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)

A comprehensive AI-powered platform for analyzing lending opportunities, predicting loan outcomes, and forecasting market trends using HMDA (Home Mortgage Disclosure Act) data and advanced machine learning techniques.

---

## ⚡ Quick Start (3 Steps)

```bash
# 1. Build and start with Docker
docker-compose up --build

# 2. Access dashboards
# Executive Dashboard: http://localhost:8501
# YoY Analysis: http://localhost:8502
```

**📖 For detailed setup and configuration instructions, see individual component documentation.**

---

## 🌟 Features

### 🎯 **Interactive Dashboards**

#### **1. Executive Dashboard** (Port 8501)
Main analytics hub with comprehensive market opportunity analysis:
- 📈 **Opportunity Analytics**: Interactive charts showing lending opportunities across census tracts
- 🗺️ **Geographic Analysis**: Interactive maps with opportunity score visualization
- 📊 **Census Tract Analysis**: Detailed temporal analysis with historical trends
- 🔮 **Temporal Forecasting**: AI-powered predictions with confidence scores
- 🎯 **Market Segmentation**: Automatic market classification
- 🤖 **Loan Prediction**: Integrated loan outcome prediction interface

#### **2. Year-over-Year Analysis Dashboard** (Port 8502)
Specialized performance analysis dashboard:
- 📊 **Market Overview**: High-level YoY trends and statistics
- 🏆 **Performance Rankings**: Top/bottom performing census tracts
- 🔍 **Tract Deep Dive**: Individual tract performance metrics
- 📈 **Market Segments**: Performance by market tier
- 💡 **Strategic Insights**: AI-generated business recommendations

#### **3. Loan Prediction System**
Standalone production-grade loan prediction module:
- 📝 **Interactive Application Form**: User-friendly loan application input
- ✅ **Real-time Predictions**: Instant approval/denial decisions
- 🎯 **Risk Assessment**: Detailed risk factor analysis
- ⚠️ **Denial Reasoning**: HMDA-compliant denial reasons
- 📊 **Model Performance**: Live model metrics and monitoring

### 🤖 **AI-Powered Analytics**
- **Opportunity Score Calculation**: AI-driven opportunity scoring for census tracts
- **Temporal Forecasting**: Future opportunity predictions with trend analysis
- **Loan Outcome Prediction**: ML models for loan approval/denial prediction
- **Market Segmentation**: Intelligent market analysis and classification
- **Denial Reason Prediction**: HMDA-compliant reasoning with explainability

### 📊 **Advanced Features**
- **Real-time Analytics**: Live data processing and visualization
- **Geographic Mapping**: Interactive maps with census tract boundaries
- **Performance Metrics**: Comprehensive model performance tracking
- **Explainable AI**: Detailed reasoning behind predictions
- **Docker Deployment**: Production-ready containerization

## 🏗️ Project Structure

```
financial-ai/
├── 📁 src/                              # Main application source code
│   ├── executive_dashboard.py           # Main executive dashboard
│   ├── enhanced_yoy_dashboard.py        # Year-over-year analysis dashboard
│   ├── comprehensive_pipeline.py        # Main pipeline orchestrator
│   ├── hmda_temporal_forecaster.py      # Temporal forecasting engine
│   ├── enhanced_loan_predictor.py       # Loan outcome prediction
│   ├── data_validator.py               # Data validation and cleaning
│   ├── opportunity_forecaster.py        # Opportunity scoring
│   ├── market_segmenter.py             # Market segmentation
│   └── enhanced_yoy_analyzer.py         # YoY analysis engine
├── � loan_prediction_system/           # Standalone loan prediction module
│   ├── src/                            # Prediction system source code
│   ├── config/                         # Model configuration
│   ├── models/                         # Trained model artifacts
│   └── Dockerfile                      # Prediction system container
├── 📁 config/                          # Configuration files
│   ├── config.yaml                     # Main configuration
│   └── config_example.yaml             # Example configuration
├── 📁 data/                            # Data directory
│   ├── actual/                         # Historical HMDA data
│   ├── generated/                      # Generated synthetic data
│   ├── models/                         # Trained model artifacts
│   └── outputs/                        # Generated results and forecasts
├── 📁 notebooks/                       # Jupyter notebooks for analysis
├── 📁 logs/                           # Application logs
├── � docker-compose.yml              # Multi-service orchestration
├── 🐳 Dockerfile                      # Main application container
├── 🐳 Dockerfile.production           # Production-optimized container
└── 📄 requirements.txt                # Python dependencies
```

## � Deployment Options

### 🐳 **Docker Deployment (Recommended)**

**Quick Start:**
```bash
# Build and run all services
docker-compose up --build

# Access dashboards:
# Executive Dashboard: http://localhost:8501
# YoY Analysis: http://localhost:8502
```

**Production Deployment:**
```bash
# Use production-optimized images
docker-compose -f docker-compose.prod.yml up -d
```

### 💻 **Local Development**

**Prerequisites:**
- Python 3.8+
- Virtual environment recommended

**Setup:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure application
cp config/config_example.yaml config/config.yaml
```

**Run Dashboards:**
```bash
# Executive Dashboard
streamlit run src/executive_dashboard.py

# YoY Analysis Dashboard  
streamlit run src/enhanced_yoy_dashboard.py --server.port 8502

# Loan Prediction System
cd loan_prediction_system
streamlit run src/dashboard.py
```

##  Core Components

### 🎯 **Comprehensive Pipeline**
The main orchestrator that runs all analysis steps including data validation, opportunity scoring, loan prediction training, temporal forecasting, and market segmentation.

### 🔮 **Temporal Forecasting**
Advanced time-series forecasting with historical analysis (2022-2024), future predictions (2025-2026), trend detection, and confidence scoring.

### 🤖 **Loan Outcome Prediction**
ML-powered loan approval/denial prediction with multiple algorithms (XGBoost, Random Forest, LightGBM, Logistic Regression) and HMDA-compliant denial reasoning.

### 📈 **Opportunity Score Calculation**
AI-driven opportunity scoring using weighted methodology:
- **Lending Activity** (30%): Volume of loan applications
- **Approval Rate** (25%): Percentage of loans approved
- **Market Accessibility** (20%): Based on loan amounts and property values
- **Economic Indicators** (15%): Income levels and stability
- **Risk Factors** (10%): Debt-to-income ratios and lending risk

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## � License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Support

For support and questions:
- 📧 Open an issue in the repository
- 📚 Check component-specific documentation
- 🔍 Review the troubleshooting section in individual READMEs

---

**🚀 Built with ❤️ for advancing fair lending analytics and opportunity identification**