#!/usr/bin/env python3
"""
Future Opportunity Score Forecasting System

Time-series forecasting for census tract-level opportunity scores using:
1. Historical lending patterns
2. Economic indicators trends  
3. Population and demographic changes
4. Housing market dynamics

Provides strategic forecasts for business planning and market entry decisions.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML Libraries for time series
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# Time series libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Configuration
# Note: Logging is configured in the main pipeline module
logger = logging.getLogger(__name__)

if not STATSMODELS_AVAILABLE:
    logger.warning("Statsmodels not available. Using ML-based forecasting only.")


class OpportunityForecaster:
    """Advanced opportunity score forecasting system"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.forecast_horizons = [1, 2, 3, 5]  # Years
        
    def create_time_series_features(self, df: pd.DataFrame, date_col: str = 'year') -> pd.DataFrame:
        """Create time-series features for forecasting"""
        
        logger.info("Creating time-series features...")
        
        # Ensure data is sorted by time
        df_ts = df.copy()
        df_ts = df_ts.sort_values([date_col, 'census_tract'])
        
        # Create lag features for each census tract
        feature_cols = [
            'opportunity_score', 'market_accessibility', 'risk_factors',
            'economic_indicators', 'lending_activity', 'total_population',
            'median_household_income', 'unemployment_rate'
        ]
        
        # Group by census tract and create features
        ts_features = []
        
        for tract in df_ts['census_tract'].unique():
            tract_data = df_ts[df_ts['census_tract'] == tract].copy()
            
            if len(tract_data) < 2:
                continue
                
            # Create lag features (1, 2, 3 periods back)
            for col in feature_cols:
                if col in tract_data.columns:
                    for lag in [1, 2, 3]:
                        tract_data[f'{col}_lag_{lag}'] = tract_data[col].shift(lag)
            
            # Create moving averages
            for col in feature_cols:
                if col in tract_data.columns:
                    tract_data[f'{col}_ma_2'] = tract_data[col].rolling(window=2, min_periods=1).mean()
                    tract_data[f'{col}_ma_3'] = tract_data[col].rolling(window=3, min_periods=1).mean()
            
            # Create trend features
            for col in feature_cols:
                if col in tract_data.columns:
                    tract_data[f'{col}_trend'] = tract_data[col].diff()
                    tract_data[f'{col}_pct_change'] = tract_data[col].pct_change()
            
            # Create seasonal features (if applicable)
            tract_data['year_sin'] = np.sin(2 * np.pi * tract_data[date_col] / 10)  # 10-year cycle
            tract_data['year_cos'] = np.cos(2 * np.pi * tract_data[date_col] / 10)
            
            ts_features.append(tract_data)
        
        if ts_features:
            result_df = pd.concat(ts_features, ignore_index=True)
            logger.info(f"Time-series features created for {len(result_df)} records")
            return result_df
        else:
            logger.warning("No time-series features could be created")
            return df_ts
    
    def simulate_historical_data(self, current_data: pd.DataFrame, years_back: int = 5) -> pd.DataFrame:
        """Simulate historical data for forecasting (when actual historical data is limited)"""
        
        logger.info(f"Simulating {years_back} years of historical data...")
        
        historical_records = []
        current_year = datetime.now().year
        
        for year in range(current_year - years_back, current_year + 1):
            year_data = current_data.copy()
            year_data['year'] = year
            
            # Add realistic variation to historical data
            if year < current_year:
                # Economic indicators tend to grow over time
                growth_factor = 1 + np.random.normal(0.02, 0.05)  # 2% average growth with variation
                year_data['median_household_income'] *= growth_factor ** (current_year - year)
                
                # Unemployment varies cyclically
                unemployment_variation = np.random.normal(0, 1.5)  # +/- 1.5% variation
                year_data['unemployment_rate'] = np.clip(
                    year_data['unemployment_rate'] + unemployment_variation, 0, 25
                )
                
                # Lending activity varies with economic cycles
                lending_variation = np.random.normal(0.95, 0.15)  # 5% average decrease with variation
                if 'loan_count' in year_data.columns:
                    year_data['loan_count'] *= lending_variation ** (current_year - year)
                
                # Recalculate opportunity scores with historical data
                from advanced_lending_platform import OpportunityScoreCalculator
                calculator = OpportunityScoreCalculator(self.config)
                year_data = calculator.calculate_opportunity_score(year_data)
            
            historical_records.append(year_data)
        
        historical_df = pd.concat(historical_records, ignore_index=True)
        logger.info(f"Historical data simulated: {len(historical_df)} records")
        
        return historical_df
    
    def prepare_forecasting_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data for forecasting models"""
        
        # Create time series features
        ts_data = self.create_time_series_features(df)
        
        # Define feature columns for forecasting
        feature_cols = []
        
        # Lag features
        base_features = ['opportunity_score', 'market_accessibility', 'risk_factors', 
                        'economic_indicators', 'lending_activity']
        
        for base in base_features:
            for lag in [1, 2, 3]:
                col = f'{base}_lag_{lag}'
                if col in ts_data.columns:
                    feature_cols.append(col)
        
        # Moving average features
        for base in base_features:
            for ma in [2, 3]:
                col = f'{base}_ma_{ma}'
                if col in ts_data.columns:
                    feature_cols.append(col)
        
        # Trend features
        for base in ['total_population', 'median_household_income', 'unemployment_rate']:
            col = f'{base}_trend'
            if col in ts_data.columns:
                feature_cols.append(col)
        
        # Seasonal features
        if 'year_sin' in ts_data.columns:
            feature_cols.extend(['year_sin', 'year_cos'])
        
        # Current economic indicators
        economic_features = ['total_population', 'median_household_income', 'unemployment_rate']
        for col in economic_features:
            if col in ts_data.columns:
                feature_cols.append(col)
        
        # Remove rows with too many missing values
        ts_data = ts_data.dropna(subset=feature_cols, thresh=len(feature_cols) * 0.7)
        
        # Fill remaining missing values
        for col in feature_cols:
            if col in ts_data.columns:
                ts_data[col] = ts_data[col].fillna(ts_data[col].median())
        
        logger.info(f"Forecasting data prepared: {len(feature_cols)} features, {len(ts_data)} records")
        
        return ts_data, feature_cols
    
    def train_forecasting_models(self, historical_data: pd.DataFrame) -> Dict:
        """Train ensemble forecasting models"""
        
        logger.info("Training opportunity score forecasting models...")
        
        # Prepare data
        forecast_data, feature_cols = self.prepare_forecasting_data(historical_data)
        
        if forecast_data.empty or not feature_cols:
            raise ValueError("Insufficient data for forecasting model training")
        
        # Prepare training data
        X = forecast_data[feature_cols].copy()
        y = forecast_data['opportunity_score'].copy()
        
        # Remove records with missing target
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Handle any remaining NaN values in features
        for col in feature_cols:
            if X[col].isna().any():
                median_val = X[col].median()
                if pd.isna(median_val):
                    X[col] = X[col].fillna(50)
                else:
                    X[col] = X[col].fillna(median_val)
        
        # Final check for NaN values
        if X.isna().any().any():
            logger.warning("Found remaining NaN values in training data, filling with 50")
            X = X.fillna(50)
        
        if len(X) < 10:
            raise ValueError("Insufficient training data for forecasting")
        
        # Scale features
        self.scalers['forecasting'] = StandardScaler()
        X_scaled = self.scalers['forecasting'].fit_transform(X)
        
        # Define models to train
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'linear': LinearRegression()
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        model_scores = {}
        
        for name, model in models.items():
            scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                scores.append(score)
            
            avg_score = np.mean(scores)
            model_scores[name] = avg_score
            logger.info(f"{name.upper()} CV R² Score: {avg_score:.4f}")
        
        # Select best model and train on full dataset
        best_model_name = max(model_scores, key=model_scores.get)
        best_model = models[best_model_name]
        best_model.fit(X_scaled, y)
        
        # Store model and features
        self.models['forecasting'] = best_model
        self.models['forecasting_features'] = feature_cols
        
        logger.info(f"Best forecasting model: {best_model_name} (R² = {model_scores[best_model_name]:.4f})")
        
        return {
            'best_model': best_model_name,
            'model_scores': model_scores,
            'feature_importance': self._get_feature_importance(best_model, feature_cols)
        }
    
    def _get_feature_importance(self, model, feature_cols: List[str]) -> Dict:
        """Get feature importance from trained model"""
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            return {}
        
        feature_importance = dict(zip(feature_cols, importance))
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def forecast_opportunity_scores(self, current_data: pd.DataFrame, 
                                   forecast_years: List[int] = None) -> pd.DataFrame:
        """Forecast future opportunity scores"""
        
        if 'forecasting' not in self.models:
            raise ValueError("Forecasting model not trained yet")
        
        if forecast_years is None:
            forecast_years = self.forecast_horizons
        
        logger.info(f"Forecasting opportunity scores for {forecast_years} years ahead...")
        
        forecasts = []
        current_year = datetime.now().year
        
        # Get the latest data for each census tract
        latest_data = current_data.groupby('census_tract').last().reset_index()
        
        for years_ahead in forecast_years:
            forecast_year = current_year + years_ahead
            
            # Create forecast data
            forecast_data = latest_data.copy()
            forecast_data['year'] = forecast_year
            
            # Apply economic growth assumptions
            growth_rates = self._get_economic_projections(years_ahead)
            
            # Project economic indicators
            forecast_data['median_household_income'] *= (1 + growth_rates['income_growth']) ** years_ahead
            forecast_data['total_population'] *= (1 + growth_rates['population_growth']) ** years_ahead
            forecast_data['unemployment_rate'] = np.clip(
                forecast_data['unemployment_rate'] * (1 + growth_rates['unemployment_change']),
                0, 25
            )
            
            # Prepare features for prediction
            # For simplicity, use current values for lag features
            feature_cols = self.models['forecasting_features']
            X_forecast = pd.DataFrame()
            
            for col in feature_cols:
                if col in forecast_data.columns:
                    X_forecast[col] = forecast_data[col]
                elif '_lag_' in col:
                    # Use current values for lag features
                    base_col = col.split('_lag_')[0]
                    if base_col in forecast_data.columns:
                        X_forecast[col] = forecast_data[base_col]
                    else:
                        X_forecast[col] = 50  # Default value
                elif '_ma_' in col:
                    # Use current values for moving averages
                    base_col = col.split('_ma_')[0]
                    if base_col in forecast_data.columns:
                        X_forecast[col] = forecast_data[base_col]
                    else:
                        X_forecast[col] = 50  # Default value
                elif '_trend' in col:
                    X_forecast[col] = 0  # Assume no trend change
                else:
                    X_forecast[col] = 50  # Default value
            
            # Fill missing columns with defaults
            for col in feature_cols:
                if col not in X_forecast.columns:
                    X_forecast[col] = 50
            
            # Ensure all feature columns are present and handle NaN values
            X_forecast = X_forecast[feature_cols].copy()
            
            # Fill any remaining NaN values with median or default
            for col in feature_cols:
                if X_forecast[col].isna().any():
                    median_val = X_forecast[col].median()
                    if pd.isna(median_val):
                        X_forecast[col] = X_forecast[col].fillna(50)
                    else:
                        X_forecast[col] = X_forecast[col].fillna(median_val)
            
            # Double check for any remaining NaN values
            if X_forecast.isna().any().any():
                logger.warning("Found remaining NaN values, filling with 50")
                X_forecast = X_forecast.fillna(50)
            
            # Scale and predict
            X_scaled = self.scalers['forecasting'].transform(X_forecast)
            predicted_scores = self.models['forecasting'].predict(X_scaled)
            
            # Create forecast dataframe
            forecast_df = forecast_data[['census_tract']].copy()
            forecast_df['forecast_year'] = forecast_year
            forecast_df['years_ahead'] = years_ahead
            forecast_df['predicted_opportunity_score'] = np.clip(predicted_scores, 0, 100)
            
            # Classify opportunity levels
            forecast_df['predicted_opportunity_level'] = forecast_df['predicted_opportunity_score'].apply(
                lambda x: 'High' if x >= 75 else 'Medium' if x >= 50 else 'Low'
            )
            
            # Add confidence intervals (simple approach)
            forecast_df['confidence_lower'] = np.clip(predicted_scores - 10, 0, 100)
            forecast_df['confidence_upper'] = np.clip(predicted_scores + 10, 0, 100)
            
            forecasts.append(forecast_df)
        
        final_forecasts = pd.concat(forecasts, ignore_index=True)
        logger.info(f"Forecasts generated for {len(final_forecasts)} census tract-year combinations")
        
        return final_forecasts
    
    def _get_economic_projections(self, years_ahead: int) -> Dict:
        """Get economic growth projections"""
        
        # Conservative economic projections
        base_rates = {
            'income_growth': 0.025,      # 2.5% annual income growth
            'population_growth': 0.008,   # 0.8% annual population growth  
            'unemployment_change': -0.02   # Slight unemployment improvement
        }
        
        # Adjust for longer-term forecasts (diminishing returns)
        adjustment_factor = 1 / (1 + years_ahead * 0.1)
        
        return {k: v * adjustment_factor for k, v in base_rates.items()}
    
    def analyze_forecast_trends(self, forecasts: pd.DataFrame) -> Dict:
        """Analyze forecast trends and insights"""
        
        logger.info("Analyzing forecast trends...")
        
        insights = {}
        
        # Overall trends
        insights['total_tracts'] = forecasts['census_tract'].nunique()
        insights['forecast_years'] = sorted(forecasts['years_ahead'].unique())
        
        # Opportunity level distribution by year
        level_trends = forecasts.groupby(['years_ahead', 'predicted_opportunity_level']).size().unstack(fill_value=0)
        insights['opportunity_level_trends'] = level_trends.to_dict()
        
        # Average score trends
        score_trends = forecasts.groupby('years_ahead')['predicted_opportunity_score'].agg(['mean', 'std']).round(2)
        insights['score_trends'] = score_trends.to_dict()
        
        # Top improving and declining tracts
        tract_changes = forecasts.pivot(index='census_tract', columns='years_ahead', 
                                      values='predicted_opportunity_score')
        
        if len(tract_changes.columns) > 1:
            first_year = min(tract_changes.columns)
            last_year = max(tract_changes.columns)
            tract_changes['total_change'] = tract_changes[last_year] - tract_changes[first_year]
            
            insights['top_improving_tracts'] = tract_changes.nlargest(5, 'total_change')[['total_change']].to_dict()
            insights['top_declining_tracts'] = tract_changes.nsmallest(5, 'total_change')[['total_change']].to_dict()
        
        return insights


def create_comprehensive_forecasts(current_data: pd.DataFrame, config) -> Tuple[pd.DataFrame, Dict]:
    """Create comprehensive opportunity score forecasts"""
    
    logger.info("Creating comprehensive opportunity score forecasts...")
    
    # Initialize forecaster
    forecaster = OpportunityForecaster(config)
    
    # Simulate historical data for training
    historical_data = forecaster.simulate_historical_data(current_data, years_back=5)
    
    # Train forecasting models
    training_results = forecaster.train_forecasting_models(historical_data)
    
    # Generate forecasts
    forecasts = forecaster.forecast_opportunity_scores(current_data)
    
    # Analyze trends
    trend_analysis = forecaster.analyze_forecast_trends(forecasts)
    
    # Combine results
    results = {
        'training_results': training_results,
        'trend_analysis': trend_analysis,
        'model_info': {
            'features_used': len(forecaster.models['forecasting_features']),
            'training_records': len(historical_data)
        }
    }
    
    return forecasts, results


if __name__ == "__main__":
    from advanced_lending_platform import LendingConfig, DataProcessor, OpportunityScoreCalculator
    
    # Initialize configuration
    config = LendingConfig()
    
    # Load current data
    processor = DataProcessor(config)
    data_sources = processor.load_all_data()
    master_data = processor.create_master_dataset()
    
    # Calculate current opportunity scores
    calculator = OpportunityScoreCalculator(config)
    current_scores = calculator.calculate_opportunity_score(master_data)
    
    # Generate forecasts
    forecasts, results = create_comprehensive_forecasts(current_scores, config)
    
    logger.info("Forecasts generated successfully")
    
    # Display summary
    print("\n" + "="*60)
    print("OPPORTUNITY SCORE FORECASTING - SUMMARY")
    print("="*60)
    print(f"Total Census Tracts: {results['trend_analysis']['total_tracts']}")
    print(f"Forecast Years: {results['trend_analysis']['forecast_years']}")
    print(f"Model Features: {results['model_info']['features_used']}")
    
    print("\nScore Trends by Year:")
    for year, stats in results['trend_analysis']['score_trends']['mean'].items():
        print(f"  Year +{year}: {stats:.1f} ± {results['trend_analysis']['score_trends']['std'][year]:.1f}")