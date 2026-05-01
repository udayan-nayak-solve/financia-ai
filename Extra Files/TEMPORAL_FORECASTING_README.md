# Temporal Opportunity Forecasting System

## 🎯 Overview

The Temporal Opportunity Forecasting System is an advanced enhancement to the Kansas Lending Opportunity Platform that provides multi-year analysis and future predictions for opportunity scores at the census tract level.

## 🚀 Key Features

### **1. Year-Based Model Training**
- **Separate Models**: Trains individual models for 2022, 2023, and 2024 data
- **Temporal Patterns**: Captures year-over-year trends and seasonality
- **Model Selection**: Automatically selects best performing model (Random Forest, XGBoost, Gradient Boosting)

### **2. Future Predictions**
- **2025 & 2026 Forecasts**: Predicts opportunity scores for upcoming years
- **Confidence Scoring**: Provides prediction confidence levels
- **Trend Analysis**: Identifies improving, declining, or stable census tracts

### **3. Enhanced Dashboard**
- **Timeline Visualization**: Shows historical vs predicted opportunity scores
- **Performance Metrics**: Displays model accuracy and performance by year
- **Strategic Insights**: Highlights consistently high-opportunity areas

## 📊 Technical Implementation

### **Data Sources**
```
data/actual/
├── 2022_state_KS.csv    # Historical lending data
├── 2023_state_KS.csv    # Historical lending data  
├── 2024_state_KS.csv    # Historical lending data
└── state_KS.csv         # Combined dataset (auto-generated)
```

### **Model Architecture**
- **Temporal Features**: Lag variables, moving averages, year-over-year changes
- **Feature Engineering**: Creates 20+ temporal features from base opportunity components
- **Ensemble Approach**: Tests multiple algorithms and selects best performer
- **Cross-Validation**: Time-series aware validation for temporal data

### **Prediction Pipeline**
1. **Data Loading**: Loads multi-year Kansas lending data
2. **Opportunity Scoring**: Calculates historical opportunity scores by year
3. **Feature Creation**: Generates temporal features and trend indicators
4. **Model Training**: Trains year-specific models with temporal features
5. **Future Prediction**: Forecasts 2025 and 2026 opportunity scores
6. **Timeline Generation**: Creates comprehensive historical vs predicted timeline

## 🔧 Usage

### **1. Data Preparation**
First, ensure you have the individual year files, then combine them:

```bash
# Combine yearly data files
python combine_data.py
```

### **2. Run Temporal Forecasting**
Test the forecasting system:

```bash
# Test temporal forecasting
python test_temporal_forecasting.py
```

### **3. Access via Dashboard**
Launch the executive dashboard and navigate to "Temporal Forecasting":

```bash
# Launch dashboard
cd src
streamlit run executive_dashboard.py
```

Select **"Temporal Forecasting"** from the navigation menu.

## 📈 Dashboard Features

### **Forecasting Summary**
- Years of historical data available
- Number of models trained
- Future predictions generated
- Total timeline records

### **Model Performance**
- R² scores by training year
- Mean Absolute Error metrics
- Best model selection results
- Performance visualization

### **Future Predictions**
- **2025 Tab**: Predicted opportunity scores for 2025
- **2026 Tab**: Predicted opportunity scores for 2026
- Distribution charts and statistics
- Top opportunity predictions

### **Timeline Analysis**
- Historical vs predicted visualization
- Trend direction indicators
- Consistently high-opportunity tracts
- Strategic recommendations

## 🎯 Business Value

### **Strategic Planning**
- **Market Entry**: Identify future high-opportunity markets
- **Resource Allocation**: Focus on areas with predicted growth
- **Risk Management**: Avoid markets with declining trends

### **Performance Tracking**
- **Trend Monitoring**: Track opportunity score evolution over time
- **Predictive Insights**: Anticipate market changes 2-3 years ahead
- **Competitive Advantage**: Data-driven strategic positioning

### **Investment Decisions**
- **Branch Placement**: Optimal locations for future expansion
- **Marketing Focus**: Target areas with improving opportunity scores
- **Portfolio Strategy**: Balance current vs future opportunity markets

## 🔍 Key Metrics

### **Model Performance**
- **R² Score**: Typically 0.65-0.80 for opportunity score prediction
- **Mean Absolute Error**: Usually 5-10 points on 0-100 scale
- **Prediction Confidence**: 70-95% confidence levels

### **Future Insights**
- **High Opportunity Tracts**: Census tracts with predicted scores ≥ 70
- **Trend Direction**: Improving, stable, or declining trajectories
- **Consistency Score**: Tracts maintaining high scores across prediction years

## 🛠️ Technical Architecture

### **Core Components**
```python
temporal_opportunity_forecaster.py    # Main forecasting engine
executive_dashboard.py                 # Enhanced dashboard with temporal views
advanced_lending_platform.py          # Base opportunity scoring system
```

### **Key Classes**
- **`TemporalOpportunityForecaster`**: Main forecasting class
- **`LendingConfig`**: Configuration management
- **`OpportunityScoreCalculator`**: Base scoring system

### **Output Files**
- **Models**: `temporal_opportunity_models_{timestamp}.pkl`
- **Predictions**: `opportunity_predictions_{timestamp}.json`
- **Logs**: Comprehensive logging of all operations

## 📚 Example Output

### **Forecasting Summary**
```
📊 Years of Historical Data: 3 years
🎯 Models Trained: 3
🔮 Future Predictions: 2 years
📈 Timeline Records: 15,420
```

### **Model Performance**
```
2022: Random Forest (R² = 0.742)
2023: XGBoost (R² = 0.768)  
2024: Gradient Boosting (R² = 0.751)
```

### **Top Future Opportunities (2025)**
```
Census Tract    Predicted Score    Confidence    Trend
20015020207          84.2            92%       Improving
20091052410          82.7            89%       Stable
20173001300          81.4            91%       Improving
```

## 🔄 Update Process

### **Quarterly Updates**
1. Add new quarterly HMDA data
2. Retrain models with updated historical data
3. Refresh future predictions
4. Update dashboard displays

### **Annual Updates**
1. Add complete new year of data
2. Extend prediction horizon (e.g., 2027, 2028)
3. Recalibrate model parameters
4. Validate prediction accuracy against actual results

## 📞 Support

For technical questions or issues with the temporal forecasting system, please refer to:
- System logs in the `logs/` directory
- Model performance metrics in the dashboard
- Configuration settings in `config/config.yaml`

---

**📈 Transform your lending strategy with predictive opportunity intelligence!**