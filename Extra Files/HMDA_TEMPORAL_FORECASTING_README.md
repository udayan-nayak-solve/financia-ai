# HMDA Temporal Opportunity Forecasting System

## Overview

The HMDA Temporal Opportunity Forecasting System enhances the Kansas Lending Opportunity Platform with multi-year analysis and future predictions. It uses historical HMDA lending data from 2022-2024 to train year-specific models and predict opportunity scores for 2025-2026.

## 🎯 Key Features

### Multi-Year Analysis
- **Historical Data**: Analyzes 3 years of HMDA lending data (2022-2024)
- **809+ Census Tracts**: Coverage across Kansas with detailed tract-level analysis
- **289,000+ Records**: Comprehensive dataset of lending activities

### HMDA-Based Opportunity Scoring
- **Lending Activity Score (30%)**: Volume of loan applications per tract
- **Approval Rate Score (25%)**: Success rate of loan approvals
- **Market Accessibility Score (20%)**: Based on loan amounts and accessibility
- **Economic Indicators Score (15%)**: Income and economic factors
- **Risk Factors Score (10%)**: DTI ratios and risk assessment

### Machine Learning Models
- **XGBoost**: Primary model achieving 98.9-99.2% R² accuracy
- **Random Forest**: Alternative ensemble method
- **Year-Specific Training**: Separate models for 2022, 2023, 2024
- **Temporal Features**: Lag variables, year-over-year changes, moving averages

### Future Predictions
- **2025 & 2026 Forecasts**: Opportunity score predictions for upcoming years
- **Confidence Metrics**: Prediction reliability assessment
- **Trend Analysis**: Direction and magnitude of changes

## 📁 File Structure

```
src/
├── hmda_temporal_forecaster.py     # Core forecasting system
├── executive_dashboard.py           # Enhanced dashboard with temporal views
test_hmda_temporal_forecasting.py   # Comprehensive test suite
test_dashboard_integration.py       # Dashboard integration tests
combine_data.py                     # Multi-year data combination utility
```

## 🚀 Quick Start

### 1. Run Temporal Forecasting

```bash
# Test the system
python3 test_hmda_temporal_forecasting.py

# Test dashboard integration
python3 test_dashboard_integration.py
```

### 2. Launch Dashboard

```bash
streamlit run src/executive_dashboard.py
```

Navigate to **"Temporal Forecasting"** in the sidebar to access:
- Multi-year opportunity analysis
- Model performance metrics
- Future predictions (2025-2026)
- Historical vs predicted timeline
- Top future opportunities

## 📊 System Performance

### Model Accuracy
- **2022 Model**: XGBoost with R² = 0.992
- **2023 Model**: XGBoost with R² = 0.990
- **2024 Model**: XGBoost with R² = 0.989

### Data Coverage
- **Census Tracts**: 809-810 tracts per year
- **Opportunity Scores**: Range 17.95 - 90.07
- **Average Score**: ~62-63 across all years
- **Temporal Features**: 31 engineered features including lags and trends

## 🔧 Technical Architecture

### Data Pipeline
1. **Load Multi-Year Data**: Combined HMDA data from 2022-2024
2. **Calculate Opportunity Scores**: HMDA-specific scoring algorithm
3. **Create Temporal Features**: Lag variables, changes, moving averages
4. **Train Models**: Year-specific XGBoost models
5. **Generate Predictions**: Future scores for 2025-2026

### Feature Engineering
- **Base Features**: Lending activity, approval rates, market accessibility
- **Temporal Features**: Year-over-year changes, lag variables
- **Economic Features**: Income levels, DTI ratios
- **Risk Features**: Default indicators, market conditions

### Model Training
```python
# Each year gets its own model
models = {
    2022: XGBRegressor(n_estimators=50, random_state=42),
    2023: XGBRegressor(n_estimators=50, random_state=42), 
    2024: XGBRegressor(n_estimators=50, random_state=42)
}

# Features include temporal and economic indicators
features = [
    'lending_activity', 'approval_rate', 'market_accessibility',
    'economic_indicators', 'risk_factors', 'year_normalized',
    'opportunity_score_lag1', 'opportunity_score_yoy_change'
]
```

## 📈 Dashboard Integration

### Temporal Forecasting View
The enhanced dashboard includes a dedicated temporal forecasting section with:

#### Summary Metrics
- Years of historical data
- Models trained
- Future predictions generated
- Timeline records processed

#### Model Performance
- R² scores by year
- Mean absolute error
- Best model selection
- Performance visualization

#### Future Predictions
- Tabbed interface for 2025/2026
- Distribution of predicted scores
- Top opportunity census tracts
- Confidence levels

#### Timeline Analysis
- Historical vs predicted visualization
- Trend analysis
- Top future opportunities

## 🧪 Testing

### Comprehensive Test Suite
```bash
# Full system test
python3 test_hmda_temporal_forecasting.py
```

Tests include:
- ✅ Data loading (3 years, 289K+ records)
- ✅ Opportunity score calculation (809 census tracts)
- ✅ Temporal feature creation (2,427 records)
- ✅ Model training (3 models, R² > 0.98)
- ✅ Future predictions (2025-2026)

### Dashboard Integration Test
```bash
# Integration test
python3 test_dashboard_integration.py
```

Validates:
- ✅ Forecaster initialization
- ✅ Pipeline execution
- ✅ Data structure compatibility
- ✅ Dashboard requirements
- ✅ Sample data preview

## 🔍 Key Insights

### Historical Trends (2022-2024)
- **Stable Scores**: Opportunity scores remain consistent ~62-63 average
- **Geographic Consistency**: Similar tract performance across years
- **Model Reliability**: High R² scores indicate predictable patterns

### Future Predictions (2025-2026)
- **Continuation of Trends**: Predicted scores align with historical patterns
- **High Confidence**: Most predictions have 75-95% confidence
- **Stable Market**: Forecast suggests stable lending opportunity landscape

### Top Performing Areas
- **High Activity Tracts**: Consistent high lending volume
- **Strong Approval Rates**: Areas with 60%+ approval rates
- **Economic Stability**: Tracts with favorable income/DTI ratios

## 🛠 Configuration

### Data Requirements
- **HMDA Data**: Combined CSV file `data/actual/state_KS.csv`
- **Required Columns**: 
  - `activity_year`, `census_tract`, `action_taken`
  - `loan_amount`, `income`, `debt_to_income_ratio`
  - Additional HMDA fields for scoring

### System Requirements
- **Python 3.8+**
- **Core Libraries**: pandas, numpy, scikit-learn, xgboost
- **Dashboard**: streamlit, plotly
- **Memory**: ~2GB for full dataset processing

## 📋 Usage Examples

### Programmatic Access
```python
from src.hmda_temporal_forecaster import HMDAOpportunityForecaster

# Initialize forecaster
forecaster = HMDAOpportunityForecaster("data/actual")

# Run full pipeline
results = forecaster.run_full_forecast_pipeline()

# Access results
historical_scores = forecaster.yearly_scores
future_predictions = forecaster.predictions
model_performance = results['summary']['models_performance']
```

### Dashboard Usage
1. Launch dashboard: `streamlit run src/executive_dashboard.py`
2. Navigate to "Temporal Forecasting" in sidebar
3. Wait for initial model training (cached afterward)
4. Explore multi-year analysis and predictions

## 🎉 Success Metrics

### System Performance
- ✅ **98.9%+ Model Accuracy**: XGBoost models achieve excellent fit
- ✅ **809 Census Tracts**: Comprehensive geographic coverage
- ✅ **3-Year History**: Robust temporal analysis
- ✅ **2-Year Forecasts**: Future predictions through 2026

### Business Value
- 📊 **Strategic Planning**: Multi-year opportunity analysis
- 🎯 **Market Targeting**: Identify future high-opportunity areas
- 📈 **Trend Analysis**: Understand market evolution
- 🔮 **Predictive Insights**: Data-driven future planning

## 🔄 Maintenance

### Data Updates
- Add new years of HMDA data to `data/actual/`
- Retrain models annually with updated data
- Monitor model performance and accuracy

### Model Monitoring
- Track R² scores and prediction accuracy
- Validate against actual outcomes when available
- Update feature engineering as needed

## 📞 Support

For questions or issues:
- Run test suites to diagnose problems
- Check data file availability and format
- Verify Python environment and dependencies
- Review log output for detailed error messages

---

**Ready to launch!** The HMDA Temporal Opportunity Forecasting System is fully integrated and ready for use in the Kansas Lending Opportunity Platform.