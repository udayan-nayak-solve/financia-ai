#!/usr/bin/env python3
"""
Streamlined Temporal Opportunity Forecasting System

A simplified version that works directly with HMDA data to:
1. Calculate opportunity scores based on HMDA lending patterns
2. Train year-specific models (2022, 2023, 2024)
3. Predict future opportunity scores (2025, 2026)
4. Provide temporal analysis for dashboard integration
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
import pickle
import json
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Configuration
# Note: Logging is configured in the main pipeline module
logger = logging.getLogger(__name__)


class HMDAOpportunityForecaster:
    """HMDA-based temporal opportunity score forecasting system"""
    
    def __init__(self, data_dir: str = None):
        """Initialize the forecaster"""
        
        self.data_dir = Path(data_dir) if data_dir else Path("data/actual")
        self.year_models = {}
        self.scalers = {}
        self.feature_columns = []
        
        # Historical and prediction years
        self.training_years = [2022, 2023, 2024]
        self.prediction_years = [2025, 2026]
        
        # Data storage
        self.historical_data = {}
        self.yearly_scores = {}
        self.predictions = {}
        
        logger.info("HMDAOpportunityForecaster initialized")
    
    def load_multi_year_data(self) -> Dict[int, pd.DataFrame]:
        """Load HMDA data for all years"""
        
        logger.info("Loading multi-year HMDA data...")
        
        yearly_data = {}
        
        # Try combined file first
        combined_file = self.data_dir / "state_KS.csv"
        
        if combined_file.exists():
            logger.info(f"Loading combined data from {combined_file}")
            
            try:
                df = pd.read_csv(combined_file, low_memory=False)
                
                for year in self.training_years:
                    year_data = df[df['activity_year'] == year].copy()
                    
                    if len(year_data) > 0:
                        yearly_data[year] = year_data
                        logger.info(f"Loaded {len(year_data):,} records for {year}")
                
            except Exception as e:
                logger.error(f"Error loading combined data: {str(e)}")
        
        self.historical_data = yearly_data
        logger.info(f"Successfully loaded data for {len(yearly_data)} years")
        
        return yearly_data
    
    def calculate_hmda_opportunity_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate opportunity scores based on HMDA lending patterns with detailed reasoning"""
        
        logger.info("Calculating HMDA-based opportunity scores...")
        
        # Group by census tract
        tract_scores = []
        
        for tract in df['census_tract'].unique():
            if pd.isna(tract):
                continue
                
            tract_data = df[df['census_tract'] == tract]
            
            if len(tract_data) < 5:  # Skip tracts with too few applications
                continue
            
            # Initialize reasoning components
            reasoning_parts = []
            component_explanations = {}
            
            # Calculate opportunity components
            score_data = {
                'census_tract': str(tract),
                'total_applications': len(tract_data),
            }
            
            # 1. Lending Activity Score (30% weight)
            # More applications = higher opportunity
            app_count = len(tract_data)
            lending_activity = min(100, (app_count / 50) * 100)  # Normalize to 0-100
            
            if app_count >= 50:
                activity_reasoning = f"High lending activity with {app_count} applications (excellent market demand)"
            elif app_count >= 25:
                activity_reasoning = f"Moderate lending activity with {app_count} applications (good market interest)"
            elif app_count >= 10:
                activity_reasoning = f"Limited lending activity with {app_count} applications (emerging market)"
            else:
                activity_reasoning = f"Low lending activity with only {app_count} applications (underdeveloped market)"
            
            component_explanations['lending_activity'] = {
                'score': round(lending_activity, 2),
                'weight': '30%',
                'reasoning': activity_reasoning
            }
            
            # 2. Approval Rate Score (25% weight)
            # Higher approval rate = higher opportunity
            approvals = len(tract_data[tract_data['action_taken'] == 1])
            approval_rate = (approvals / len(tract_data)) * 100 if len(tract_data) > 0 else 0
            
            if approval_rate >= 80:
                approval_reasoning = f"Excellent approval rate of {approval_rate:.1f}% indicates strong creditworthiness"
            elif approval_rate >= 60:
                approval_reasoning = f"Good approval rate of {approval_rate:.1f}% shows moderate lending success"
            elif approval_rate >= 40:
                approval_reasoning = f"Fair approval rate of {approval_rate:.1f}% suggests some lending challenges"
            else:
                approval_reasoning = f"Low approval rate of {approval_rate:.1f}% indicates significant lending barriers"
            
            component_explanations['approval_rate'] = {
                'score': round(approval_rate, 2),
                'weight': '25%',
                'reasoning': approval_reasoning
            }
            
            # 3. Market Accessibility Score (20% weight)
            # Based on loan amounts and property values
            if 'loan_amount' in tract_data.columns:
                avg_loan_amount = tract_data['loan_amount'].median()
                # Normalize loan amounts (higher amounts = more accessible capital)
                market_accessibility = min(100, (avg_loan_amount / 500000) * 100)
                
                if avg_loan_amount >= 400000:
                    access_reasoning = f"High-value market with median loan of ${avg_loan_amount:,.0f} (premium segment)"
                elif avg_loan_amount >= 250000:
                    access_reasoning = f"Mid-range market with median loan of ${avg_loan_amount:,.0f} (mainstream segment)"
                elif avg_loan_amount >= 150000:
                    access_reasoning = f"Affordable market with median loan of ${avg_loan_amount:,.0f} (entry-level segment)"
                else:
                    access_reasoning = f"Low-value market with median loan of ${avg_loan_amount:,.0f} (budget segment)"
            else:
                market_accessibility = 50  # Default
                access_reasoning = "Market accessibility based on standard lending patterns (limited loan data)"
            
            component_explanations['market_accessibility'] = {
                'score': round(market_accessibility, 2),
                'weight': '20%',
                'reasoning': access_reasoning
            }
            
            # 4. Economic Indicators Score (15% weight)
            # Based on income and property values
            economic_score = 50  # Default
            if 'income' in tract_data.columns:
                avg_income = tract_data['income'].median()
                economic_score = min(100, (avg_income / 150) * 100)  # Assuming income in thousands
                
                if avg_income >= 120:
                    income_reasoning = f"High-income area with median income ${avg_income:.0f}k (strong economic foundation)"
                elif avg_income >= 80:
                    income_reasoning = f"Middle-income area with median income ${avg_income:.0f}k (stable economic base)"
                elif avg_income >= 50:
                    income_reasoning = f"Moderate-income area with median income ${avg_income:.0f}k (developing economic base)"
                else:
                    income_reasoning = f"Low-income area with median income ${avg_income:.0f}k (economic growth potential)"
            else:
                income_reasoning = "Economic indicators based on regional averages (limited income data)"
            
            component_explanations['economic_indicators'] = {
                'score': round(economic_score, 2),
                'weight': '15%',
                'reasoning': income_reasoning
            }
            
            # 5. Risk Factors Score (10% weight)
            # Lower DTI and higher property values = lower risk = higher opportunity
            risk_score = 50  # Default
            calculated_dti = 0
            if 'debt_to_income_ratio' in tract_data.columns:
                # Convert DTI categories to numeric midpoints
                dti_numeric = []
                for dti in tract_data['debt_to_income_ratio']:
                    if pd.isna(dti) or dti == 'Exempt':
                        continue
                    elif isinstance(dti, str):
                        if '<20%' in dti:
                            dti_numeric.append(15)
                        elif '20%-<30%' in dti:
                            dti_numeric.append(25)
                        elif '30%-<36%' in dti:
                            dti_numeric.append(33)
                        elif '50%-60%' in dti:
                            dti_numeric.append(55)
                        elif '>60%' in dti:
                            dti_numeric.append(70)
                    else:
                        try:
                            dti_numeric.append(float(dti))
                        except:
                            continue
                
                if dti_numeric:
                    calculated_dti = np.median(dti_numeric)
                    # Lower DTI = higher score (inverse relationship)
                    risk_score = max(0, 100 - (calculated_dti * 1.5))  # DTI of 67% = 0 score
                    
                    if calculated_dti <= 25:
                        risk_reasoning = f"Low risk profile with median DTI of {calculated_dti:.1f}% (excellent financial health)"
                    elif calculated_dti <= 35:
                        risk_reasoning = f"Moderate risk profile with median DTI of {calculated_dti:.1f}% (good financial stability)"
                    elif calculated_dti <= 50:
                        risk_reasoning = f"Higher risk profile with median DTI of {calculated_dti:.1f}% (manageable debt levels)"
                    else:
                        risk_reasoning = f"High risk profile with median DTI of {calculated_dti:.1f}% (elevated debt burden)"
                else:
                    risk_reasoning = "Risk assessment based on standard lending criteria (limited DTI data)"
            else:
                risk_reasoning = "Risk factors evaluated using regional lending patterns (no DTI data available)"
            
            component_explanations['risk_factors'] = {
                'score': round(risk_score, 2),
                'weight': '10%',
                'reasoning': risk_reasoning
            }
            
            # Calculate weighted opportunity score
            opportunity_score = (
                lending_activity * 0.30 +
                approval_rate * 0.25 +
                market_accessibility * 0.20 +
                economic_score * 0.15 +
                risk_score * 0.10
            )
            
            # Generate overall reasoning summary
            if opportunity_score >= 80:
                overall_assessment = "Excellent opportunity zone with strong fundamentals across all metrics"
            elif opportunity_score >= 65:
                overall_assessment = "Good opportunity zone with solid performance in most areas"
            elif opportunity_score >= 50:
                overall_assessment = "Moderate opportunity zone with balanced risk-reward profile"
            elif opportunity_score >= 35:
                overall_assessment = "Emerging opportunity zone with potential for growth"
            else:
                overall_assessment = "Developing opportunity zone requiring strategic focus"
            
            # Create detailed reasoning summary
            reasoning_summary = {
                'overall_score': round(opportunity_score, 2),
                'assessment': overall_assessment,
                'components': component_explanations,
                'key_factors': [
                    f"Lending activity contributes {round(lending_activity * 0.30, 1)} points (30% weight)",
                    f"Approval rate contributes {round(approval_rate * 0.25, 1)} points (25% weight)",
                    f"Market accessibility contributes {round(market_accessibility * 0.20, 1)} points (20% weight)",
                    f"Economic indicators contribute {round(economic_score * 0.15, 1)} points (15% weight)",
                    f"Risk factors contribute {round(risk_score * 0.10, 1)} points (10% weight)"
                ],
                'calculation_method': 'Weighted average of five key lending opportunity components'
            }
            
            # Store component scores and reasoning
            score_data.update({
                'opportunity_score': round(opportunity_score, 2),
                'lending_activity': round(lending_activity, 2),
                'approval_rate': round(approval_rate, 2),
                'market_accessibility': round(market_accessibility, 2),
                'economic_indicators': round(economic_score, 2),
                'risk_factors': round(risk_score, 2),
                'avg_loan_amount': tract_data['loan_amount'].median() if 'loan_amount' in tract_data.columns else 0,
                'avg_income': tract_data['income'].median() if 'income' in tract_data.columns else 0,
                'avg_dti': calculated_dti,
                'reasoning': reasoning_summary  # Add detailed reasoning
            })
            
            tract_scores.append(score_data)
        
        # Convert to DataFrame
        scores_df = pd.DataFrame(tract_scores)
        
        logger.info(f"Calculated opportunity scores for {len(scores_df)} census tracts")
        
        return scores_df
    
    def calculate_yearly_opportunity_scores(self) -> Dict[int, pd.DataFrame]:
        """Calculate opportunity scores for each year"""
        
        logger.info("Calculating opportunity scores by year...")
        
        yearly_scores = {}
        
        for year, data in self.historical_data.items():
            logger.info(f"Processing opportunity scores for {year}...")
            
            scored_data = self.calculate_hmda_opportunity_scores(data)
            scored_data['year'] = year
            
            yearly_scores[year] = scored_data
            
            logger.info(f"Calculated scores for {len(scored_data)} census tracts in {year}")
        
        self.yearly_scores = yearly_scores
        return yearly_scores
    
    def create_temporal_features(self) -> pd.DataFrame:
        """Create temporal features for forecasting"""
        
        logger.info("Creating temporal features...")
        
        # Combine all yearly data
        combined_data = []
        
        for year, data in self.yearly_scores.items():
            data_copy = data.copy()
            combined_data.append(data_copy)
        
        if not combined_data:
            raise ValueError("No yearly scores available")
        
        # Combine all years
        full_data = pd.concat(combined_data, ignore_index=True)
        full_data = full_data.sort_values(['census_tract', 'year'])
        
        # Create temporal features
        temporal_data = []
        
        for tract in full_data['census_tract'].unique():
            tract_data = full_data[full_data['census_tract'] == tract].copy()
            
            if len(tract_data) < 2:
                continue
            
            # Add temporal features
            feature_cols = [
                'opportunity_score', 'lending_activity', 'approval_rate',
                'market_accessibility', 'economic_indicators', 'risk_factors'
            ]
            
            for col in feature_cols:
                if col in tract_data.columns:
                    # Lag features
                    tract_data[f'{col}_lag1'] = tract_data[col].shift(1)
                    
                    # Year-over-year change
                    tract_data[f'{col}_yoy_change'] = tract_data[col].diff()
                    tract_data[f'{col}_yoy_pct_change'] = tract_data[col].pct_change()
                    
                    # Moving average
                    tract_data[f'{col}_ma2'] = tract_data[col].rolling(window=2, min_periods=1).mean()
            
            # Time features
            tract_data['year_normalized'] = (tract_data['year'] - 2022) / 2  # 0 to 1 for 2022-2024
            
            temporal_data.append(tract_data)
        
        if not temporal_data:
            raise ValueError("No temporal features created")
        
        final_data = pd.concat(temporal_data, ignore_index=True)
        final_data = final_data.fillna(method='ffill').fillna(0)
        
        # Replace infinite values with finite values
        final_data = final_data.replace([np.inf, -np.inf], 0)
        
        # Ensure all numeric columns are finite
        numeric_cols = final_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            final_data[col] = np.where(np.isfinite(final_data[col]), final_data[col], 0)
        
        logger.info(f"Created temporal features for {len(final_data)} records")
        
        return final_data
    
    def train_temporal_models(self, temporal_data: pd.DataFrame) -> Dict[int, Dict]:
        """Train temporal forecasting models"""
        
        logger.info("Training temporal forecasting models...")
        
        # Define features
        base_features = [
            'lending_activity', 'approval_rate', 'market_accessibility',
            'economic_indicators', 'risk_factors', 'year_normalized',
            'total_applications'  # Add this as a base feature
        ]
        
        # Add lag and change features
        lag_features = [col for col in temporal_data.columns if '_lag1' in col]
        change_features = [col for col in temporal_data.columns if '_yoy_change' in col or '_yoy_pct_change' in col]
        ma_features = [col for col in temporal_data.columns if '_ma2' in col]
        
        # Combine all features and filter for available ones
        all_features = base_features + lag_features + change_features + ma_features
        available_features = [col for col in all_features if col in temporal_data.columns]
        
        # If no lag/change features available, use just base features
        if len(available_features) < 3:
            available_features = [col for col in base_features if col in temporal_data.columns]
        
        self.feature_columns = available_features
        
        logger.info(f"Using {len(available_features)} features: {available_features[:5]}...")
        
        year_models = {}
        
        for year in self.training_years:
            logger.info(f"Training model for {year}...")
            
            # Get data up to this year
            train_data = temporal_data[temporal_data['year'] <= year].copy()
            
            if len(train_data) < 20:
                logger.warning(f"Insufficient data for {year}")
                continue
            
            # Prepare features and target
            X = train_data[available_features].fillna(0)
            y = train_data['opportunity_score']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            models = {
                'random_forest': RandomForestRegressor(n_estimators=50, random_state=42),
                'xgboost': xgb.XGBRegressor(n_estimators=50, random_state=42)
            }
            
            best_model = None
            best_score = -np.inf
            model_results = {}
            
            for model_name, model in models.items():
                if model_name == 'random_forest':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                model_results[model_name] = {
                    'model': model,
                    'r2_score': r2,
                    'mse': mse,
                    'mae': mae
                }
                
                if r2 > best_score:
                    best_score = r2
                    best_model = model_name
            
            year_models[year] = {
                'best_model': best_model,
                'models': model_results,
                'scaler': scaler,
                'performance': model_results[best_model]
            }
            
            logger.info(f"Best model for {year}: {best_model} (R² = {best_score:.3f})")
        
        self.year_models = year_models
        return year_models
    
    def predict_future_scores(self, temporal_data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """Predict opportunity scores for future years with temporal progression"""
        
        logger.info("Predicting future opportunity scores with temporal analysis...")
        
        predictions = {}
        
        # Use most recent year's model
        recent_year = max(self.training_years)
        if recent_year not in self.year_models:
            logger.error("No trained model available")
            return predictions
        
        model_info = self.year_models[recent_year]
        best_model = model_info['models'][model_info['best_model']]['model']
        scaler = model_info['scaler']
        
        # Calculate temporal trends for each census tract
        logger.info("Analyzing temporal trends...")
        tract_trends = self._calculate_tract_trends(temporal_data)
        
        # Get base data for predictions (most recent complete data)
        base_data = temporal_data[temporal_data['year'] == recent_year].copy()
        
        # For each prediction year, apply temporal progression
        for pred_year in self.prediction_years:
            logger.info(f"Predicting scores for {pred_year} with {pred_year - recent_year} year progression...")
            
            # Calculate years ahead for progression
            years_ahead = pred_year - recent_year
            
            # Create future data with temporal progression
            future_data = self._apply_temporal_progression(
                base_data, tract_trends, pred_year, years_ahead
            )
            
            # Prepare features for prediction
            X_future = future_data[self.feature_columns].fillna(0)
            
            # Make predictions
            if model_info['best_model'] == 'random_forest':
                X_future_scaled = scaler.transform(X_future)
                predicted_scores = best_model.predict(X_future_scaled)
            else:
                predicted_scores = best_model.predict(X_future)
            
            # Create prediction dataframe with trend analysis
            pred_df = future_data[['census_tract']].copy()
            pred_df['predicted_opportunity_score'] = np.clip(predicted_scores, 0, 100)
            
            # Add trend information and confidence
            pred_df = pred_df.merge(tract_trends[['census_tract', 'trend_direction', 'avg_yoy_change', 'trend_strength']], 
                                   on='census_tract', how='left')
            
            # Calculate prediction confidence based on trend consistency
            pred_df['prediction_confidence'] = np.clip(
                85 - (years_ahead * 5) + (pred_df['trend_strength'] * 10), 60, 95
            )
            
            predictions[pred_year] = pred_df
            
            logger.info(f"Generated predictions for {len(pred_df)} census tracts for {pred_year}")
            logger.info(f"Average predicted score: {pred_df['predicted_opportunity_score'].mean():.2f}")
        
        self.predictions = predictions
        return predictions
    
    def _calculate_tract_trends(self, temporal_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend analysis for each census tract"""
        
        trend_results = []
        
        for tract in temporal_data['census_tract'].unique():
            tract_data = temporal_data[temporal_data['census_tract'] == tract].copy()
            tract_data = tract_data.sort_values('year')
            
            if len(tract_data) < 2:
                # Not enough data for trend analysis
                trend_results.append({
                    'census_tract': tract,
                    'trend_direction': 'stable',
                    'avg_yoy_change': 0,
                    'trend_strength': 0.5,
                    'volatility': 0
                })
                continue
            
            # Calculate year-over-year changes in opportunity score
            scores = tract_data['opportunity_score'].values
            yoy_changes = np.diff(scores)
            
            # Calculate trend metrics
            avg_yoy_change = np.mean(yoy_changes) if len(yoy_changes) > 0 else 0
            volatility = np.std(yoy_changes) if len(yoy_changes) > 1 else 0
            
            # Determine trend direction
            if avg_yoy_change > 2:
                trend_direction = 'increasing'
            elif avg_yoy_change < -2:
                trend_direction = 'decreasing'
            else:
                trend_direction = 'stable'
            
            # Calculate trend strength (consistency)
            if len(yoy_changes) > 1:
                # Trend strength based on consistency of direction
                positive_changes = sum(1 for x in yoy_changes if x > 0)
                negative_changes = sum(1 for x in yoy_changes if x < 0)
                total_changes = len(yoy_changes)
                
                if total_changes > 0:
                    trend_strength = max(positive_changes, negative_changes) / total_changes
                else:
                    trend_strength = 0.5
            else:
                trend_strength = 0.5
            
            trend_results.append({
                'census_tract': tract,
                'trend_direction': trend_direction,
                'avg_yoy_change': avg_yoy_change,
                'trend_strength': trend_strength,
                'volatility': volatility
            })
        
        return pd.DataFrame(trend_results)
    
    def _apply_temporal_progression(self, base_data: pd.DataFrame, tract_trends: pd.DataFrame, 
                                   pred_year: int, years_ahead: int) -> pd.DataFrame:
        """Apply temporal progression to base data for future predictions"""
        
        future_data = base_data.copy()
        future_data['year'] = pred_year
        future_data['year_normalized'] = (pred_year - 2022) / 4  # Normalize over extended range
        
        # Merge with trend data
        future_data = future_data.merge(tract_trends, on='census_tract', how='left')
        
        # Apply temporal progression to key features
        progression_features = [
            'lending_activity', 'approval_rate', 'market_accessibility',
            'economic_indicators', 'opportunity_score'
        ]
        
        for feature in progression_features:
            if feature in future_data.columns:
                # Apply trend-based progression
                base_values = future_data[feature].fillna(0)
                avg_change = future_data['avg_yoy_change'].fillna(0)
                
                # Calculate progression factor based on years ahead and trend
                # Dampen the effect over time to avoid unrealistic projections
                damping_factor = 0.8 ** years_ahead  # Exponential dampening
                progression = avg_change * years_ahead * damping_factor
                
                # Apply progression with bounds checking
                if feature == 'opportunity_score':
                    # For opportunity score, apply the calculated trend
                    future_data[feature] = np.clip(base_values + progression, 0, 100)
                elif feature in ['approval_rate']:
                    # For rates, apply smaller progression and keep in valid range
                    future_data[feature] = np.clip(base_values + (progression * 0.5), 0, 100)
                else:
                    # For other features, apply moderate progression
                    future_data[feature] = np.clip(base_values + (progression * 0.7), 0, 
                                                 base_values.max() * 1.5)
        
        # Update lag features with adjusted values
        lag_features = [col for col in future_data.columns if '_lag1' in col]
        for lag_feature in lag_features:
            base_feature = lag_feature.replace('_lag1', '')
            if base_feature in future_data.columns:
                future_data[lag_feature] = future_data[base_feature]
        
        # Update year-over-year change features
        yoy_features = [col for col in future_data.columns if '_yoy_change' in col]
        for yoy_feature in yoy_features:
            base_feature = yoy_feature.replace('_yoy_change', '')
            if base_feature in future_data.columns:
                future_data[yoy_feature] = future_data['avg_yoy_change'] * damping_factor
        
        # Update moving averages
        ma_features = [col for col in future_data.columns if '_ma2' in col]
        for ma_feature in ma_features:
            base_feature = ma_feature.replace('_ma2', '')
            if base_feature in future_data.columns:
                future_data[ma_feature] = future_data[base_feature]
        
        # Clean up temporary columns
        future_data = future_data.drop(columns=['trend_direction', 'avg_yoy_change', 'trend_strength', 'volatility'], 
                                     errors='ignore')
        
        return future_data
    
    def run_full_forecast_pipeline(self) -> Dict:
        """Run the complete forecasting pipeline"""
        
        logger.info("Starting HMDA temporal forecasting pipeline...")
        
        try:
            # Load data
            yearly_data = self.load_multi_year_data()
            
            if not yearly_data:
                raise ValueError("No yearly data loaded")
            
            # Calculate opportunity scores
            yearly_scores = self.calculate_yearly_opportunity_scores()
            
            # Create temporal features
            temporal_data = self.create_temporal_features()
            
            # Train models
            year_models = self.train_temporal_models(temporal_data)
            
            # Predict future scores
            predictions = self.predict_future_scores(temporal_data)
            
            # Compile results
            results = {
                'success': True,
                'yearly_data_loaded': len(yearly_data),
                'models_trained': len(year_models),
                'predictions_generated': len(predictions),
                'timeline_records': len(temporal_data),
                'summary': {
                    'training_years': self.training_years,
                    'prediction_years': self.prediction_years,
                    'models_performance': {
                        str(year): {
                            'best_model': info['best_model'],
                            'r2_score': round(info['performance']['r2_score'], 3),
                            'mae': round(info['performance']['mae'], 3)
                        }
                        for year, info in year_models.items()
                    },
                    'prediction_summary': {
                        str(year): {
                            'census_tracts': len(pred_data),
                            'avg_predicted_score': round(pred_data['predicted_opportunity_score'].mean(), 2),
                            'score_range': [
                                round(pred_data['predicted_opportunity_score'].min(), 2),
                                round(pred_data['predicted_opportunity_score'].max(), 2)
                            ]
                        }
                        for year, pred_data in predictions.items()
                    }
                }
            }
            
            logger.info("HMDA temporal forecasting pipeline completed successfully!")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in forecasting pipeline: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_pipeline(self) -> Dict:
        """Wrapper method for compatibility with comprehensive pipeline"""
        return self.run_full_forecast_pipeline()


def create_hmda_temporal_forecasts(data_dir: str = None) -> Dict:
    """Main function to create HMDA-based temporal forecasts"""
    
    logger.info("Creating HMDA temporal opportunity forecasts...")
    
    forecaster = HMDAOpportunityForecaster(data_dir)
    results = forecaster.run_full_forecast_pipeline()
    
    return results


if __name__ == "__main__":
    results = create_hmda_temporal_forecasts()
    
    if results['success']:
        print("✅ HMDA Temporal forecasting completed successfully!")
        print(f"📊 Summary: {results['summary']}")
    else:
        print(f"❌ Forecasting failed: {results['error']}")