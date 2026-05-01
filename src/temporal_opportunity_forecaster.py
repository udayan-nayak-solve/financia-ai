#!/usr/bin/env python3
"""
Temporal Opportunity Score Forecasting System

Advanced multi-year opportunity score forecasting with:
1. Year-based model training (2022, 2023, 2024)
2. Future predictions (2025, 2026)
3. Temporal pattern analysis
4. Trend-based insights for strategic planning

Designed for executive dashboard integration.
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

# Custom modules
from advanced_lending_platform import LendingConfig, DataProcessor, OpportunityScoreCalculator

# Configuration
# Note: Logging is configured in the main pipeline module
logger = logging.getLogger(__name__)


class TemporalOpportunityForecaster:
    """Advanced temporal opportunity score forecasting system"""
    
    def __init__(self, config: LendingConfig = None):
        """Initialize the temporal forecasting system"""
        
        self.config = config or LendingConfig()
        self.year_models = {}  # Models trained for each year
        self.scalers = {}
        self.feature_columns = []
        self.temporal_features = []
        
        # Historical and prediction years
        self.training_years = [2022, 2023, 2024]
        self.prediction_years = [2025, 2026]
        
        # Data storage
        self.historical_data = {}
        self.predictions = {}
        
        logger.info("TemporalOpportunityForecaster initialized")
    
    def load_multi_year_data(self) -> Dict[int, pd.DataFrame]:
        """Load and process data for all available years"""
        
        logger.info("Loading multi-year Kansas lending data...")
        
        yearly_data = {}
        
        # Try to load combined file first
        combined_file = self.config.DATA_DIR / "state_KS.csv"
        
        if combined_file.exists():
            logger.info(f"Loading combined data from {combined_file}")
            
            try:
                # Load combined data
                df = pd.read_csv(combined_file, low_memory=False)
                
                # Split by year
                for year in self.training_years:
                    year_data = df[df['activity_year'] == year].copy()
                    
                    if len(year_data) > 0:
                        yearly_data[year] = year_data
                        logger.info(f"Loaded {len(year_data):,} records for {year}")
                    else:
                        logger.warning(f"No data found for {year}")
                
            except Exception as e:
                logger.error(f"Error loading combined data: {str(e)}")
        
        else:
            # Fallback to individual files
            for year in self.training_years:
                try:
                    # Load year-specific data
                    file_path = self.config.DATA_DIR / f"{year}_state_KS.csv"
                    
                    if file_path.exists():
                        logger.info(f"Loading {year} data from {file_path}")
                        
                        # Load data
                        df = pd.read_csv(file_path, low_memory=False)
                        
                        # Add year column if not present
                        if 'activity_year' not in df.columns:
                            df['activity_year'] = year
                        
                        yearly_data[year] = df
                        logger.info(f"Loaded {len(df):,} records for {year}")
                        
                    else:
                        logger.warning(f"Data file not found for {year}: {file_path}")
                        
                except Exception as e:
                    logger.error(f"Error loading {year} data: {str(e)}")
        
        self.historical_data = yearly_data
        logger.info(f"Successfully loaded data for {len(yearly_data)} years")
        
        return yearly_data
    
    def calculate_yearly_opportunity_scores(self) -> Dict[int, pd.DataFrame]:
        """Calculate opportunity scores for each year separately"""
        
        logger.info("Calculating opportunity scores by year...")
        
        yearly_scores = {}
        
        for year, data in self.historical_data.items():
            try:
                logger.info(f"Processing opportunity scores for {year}...")
                
                # Initialize opportunity calculator
                calculator = OpportunityScoreCalculator(self.config)
                
                # Calculate scores for this year
                scored_data = calculator.calculate_opportunity_score(data)
                
                # Add year and create unique identifier
                scored_data['year'] = year
                scored_data['tract_year_id'] = scored_data['census_tract'].astype(str) + f"_{year}"
                
                yearly_scores[year] = scored_data
                
                logger.info(f"Calculated scores for {len(scored_data)} census tracts in {year}")
                
            except Exception as e:
                logger.error(f"Error calculating scores for {year}: {str(e)}")
        
        return yearly_scores
    
    def create_temporal_features(self, yearly_scores: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        """Create temporal features for forecasting"""
        
        logger.info("Creating temporal features for forecasting...")
        
        # Combine all yearly data
        combined_data = []
        
        for year, data in yearly_scores.items():
            data_copy = data.copy()
            data_copy['year'] = year
            combined_data.append(data_copy)
        
        if not combined_data:
            raise ValueError("No yearly data available for temporal feature creation")
        
        # Combine all years
        full_data = pd.concat(combined_data, ignore_index=True)
        
        # Sort by census tract and year
        full_data = full_data.sort_values(['census_tract', 'year'])
        
        # Create temporal features
        temporal_data = []
        
        for tract in full_data['census_tract'].unique():
            tract_data = full_data[full_data['census_tract'] == tract].copy()
            
            if len(tract_data) < 2:  # Need at least 2 years for trends
                continue
            
            # Add temporal features
            feature_cols = [
                'opportunity_score', 'market_accessibility', 'risk_factors',
                'economic_indicators', 'lending_activity'
            ]
            
            for col in feature_cols:
                if col in tract_data.columns:
                    # Lag features (previous year values)
                    tract_data[f'{col}_lag1'] = tract_data[col].shift(1)
                    tract_data[f'{col}_lag2'] = tract_data[col].shift(2)
                    
                    # Trend features (year-over-year change)
                    tract_data[f'{col}_yoy_change'] = tract_data[col].diff()
                    tract_data[f'{col}_yoy_pct_change'] = tract_data[col].pct_change()
                    
                    # Moving averages
                    tract_data[f'{col}_ma2'] = tract_data[col].rolling(window=2, min_periods=1).mean()
                    tract_data[f'{col}_ma3'] = tract_data[col].rolling(window=3, min_periods=1).mean()
            
            # Time-based features
            tract_data['year_normalized'] = (tract_data['year'] - tract_data['year'].min()) / (tract_data['year'].max() - tract_data['year'].min())
            tract_data['year_sin'] = np.sin(2 * np.pi * tract_data['year'] / 10)  # 10-year cycle
            tract_data['year_cos'] = np.cos(2 * np.pi * tract_data['year'] / 10)
            
            temporal_data.append(tract_data)
        
        if not temporal_data:
            raise ValueError("No temporal features could be created")
        
        # Combine all temporal data
        final_data = pd.concat(temporal_data, ignore_index=True)
        
        # Fill NaN values with forward fill then backward fill
        final_data = final_data.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Created temporal features for {len(final_data)} records")
        
        return final_data
    
    def prepare_training_data(self, temporal_data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for model training"""
        
        logger.info("Preparing training data...")
        
        # Define feature columns
        base_features = [
            'market_accessibility', 'risk_factors', 'economic_indicators', 
            'lending_activity', 'year_normalized', 'year_sin', 'year_cos'
        ]
        
        # Add lag features
        lag_features = [col for col in temporal_data.columns if '_lag1' in col or '_lag2' in col]
        
        # Add trend features
        trend_features = [col for col in temporal_data.columns if '_yoy_change' in col or '_yoy_pct_change' in col]
        
        # Add moving average features
        ma_features = [col for col in temporal_data.columns if '_ma2' in col or '_ma3' in col]
        
        # Combine all features
        all_features = base_features + lag_features + trend_features + ma_features
        
        # Filter to available columns
        available_features = [col for col in all_features if col in temporal_data.columns]
        
        # Create feature matrix
        feature_data = temporal_data[available_features + ['opportunity_score', 'year', 'census_tract']].copy()
        
        # Remove rows with missing values
        feature_data = feature_data.dropna()
        
        self.feature_columns = available_features
        
        logger.info(f"Prepared {len(available_features)} features for {len(feature_data)} records")
        
        return feature_data, available_features
    
    def train_yearly_models(self, training_data: pd.DataFrame) -> Dict[int, Dict]:
        """Train separate models for each year"""
        
        logger.info("Training year-specific opportunity score models...")
        
        year_models = {}
        
        for year in self.training_years:
            try:
                logger.info(f"Training model for {year}...")
                
                # Get data for this year and previous years (for temporal context)
                year_data = training_data[training_data['year'] <= year].copy()
                
                if len(year_data) < 10:  # Need minimum data for training
                    logger.warning(f"Insufficient data for {year} model training")
                    continue
                
                # Prepare features and target
                X = year_data[self.feature_columns]
                y = year_data['opportunity_score']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train multiple models
                models = {
                    'random_forest': RandomForestRegressor(
                        n_estimators=100, 
                        max_depth=15, 
                        random_state=42,
                        n_jobs=-1
                    ),
                    'gradient_boosting': GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        random_state=42
                    ),
                    'xgboost': xgb.XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        random_state=42
                    )
                }
                
                best_model = None
                best_score = -np.inf
                model_results = {}
                
                for model_name, model in models.items():
                    # Train model
                    if model_name in ['random_forest', 'gradient_boosting']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:  # XGBoost
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Evaluate
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
                
                # Store the best model for this year
                year_models[year] = {
                    'best_model': best_model,
                    'models': model_results,
                    'scaler': scaler,
                    'performance': model_results[best_model],
                    'training_size': len(X_train)
                }
                
                logger.info(f"Best model for {year}: {best_model} (R² = {best_score:.3f})")
                
            except Exception as e:
                logger.error(f"Error training model for {year}: {str(e)}")
        
        self.year_models = year_models
        logger.info(f"Successfully trained models for {len(year_models)} years")
        
        return year_models
    
    def predict_future_scores(self, temporal_data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """Predict opportunity scores for future years"""
        
        logger.info("Predicting future opportunity scores...")
        
        predictions = {}
        
        # Get the most recent year's data as baseline
        recent_year = max(self.training_years)
        recent_data = temporal_data[temporal_data['year'] == recent_year].copy()
        
        if recent_data.empty:
            logger.error("No recent data available for predictions")
            return predictions
        
        # Use the best performing model from recent years
        if recent_year not in self.year_models:
            logger.error(f"No trained model available for base year {recent_year}")
            return predictions
        
        model_info = self.year_models[recent_year]
        best_model = model_info['models'][model_info['best_model']]['model']
        scaler = model_info['scaler']
        
        for pred_year in self.prediction_years:
            try:
                logger.info(f"Predicting scores for {pred_year}...")
                
                # Create future data by extrapolating trends
                future_data = recent_data.copy()
                future_data['year'] = pred_year
                
                # Update time-based features
                future_data['year_normalized'] = (pred_year - min(self.training_years)) / (max(self.training_years) - min(self.training_years))
                future_data['year_sin'] = np.sin(2 * np.pi * pred_year / 10)
                future_data['year_cos'] = np.cos(2 * np.pi * pred_year / 10)
                
                # Extrapolate trend features (simple linear extrapolation)
                years_ahead = pred_year - recent_year
                
                for col in self.feature_columns:
                    if '_yoy_change' in col:
                        # Apply the trend
                        base_col = col.replace('_yoy_change', '')
                        if base_col in future_data.columns:
                            trend_value = future_data[col].fillna(0)
                            future_data[base_col] = future_data[base_col] + (trend_value * years_ahead)
                
                # Prepare features for prediction
                X_future = future_data[self.feature_columns]
                
                # Handle missing values
                X_future = X_future.fillna(X_future.mean())
                
                # Make predictions
                if model_info['best_model'] in ['random_forest', 'gradient_boosting']:
                    X_future_scaled = scaler.transform(X_future)
                    predicted_scores = best_model.predict(X_future_scaled)
                else:  # XGBoost
                    predicted_scores = best_model.predict(X_future)
                
                # Create prediction dataframe
                pred_df = future_data[['census_tract']].copy()
                pred_df['year'] = pred_year
                pred_df['predicted_opportunity_score'] = predicted_scores
                pred_df['prediction_confidence'] = self._calculate_prediction_confidence(predicted_scores)
                pred_df['trend_direction'] = self._determine_trend_direction(future_data, recent_data)
                
                predictions[pred_year] = pred_df
                
                logger.info(f"Generated predictions for {len(pred_df)} census tracts for {pred_year}")
                
            except Exception as e:
                logger.error(f"Error predicting scores for {pred_year}: {str(e)}")
        
        self.predictions = predictions
        return predictions
    
    def _calculate_prediction_confidence(self, scores: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for predictions"""
        
        # Simple confidence based on score stability
        # Higher scores and mid-range scores get higher confidence
        confidence = np.ones_like(scores)
        
        # Reduce confidence for extreme values
        confidence[scores < 20] *= 0.7  # Low scores less confident
        confidence[scores > 80] *= 0.8  # Very high scores less confident
        
        # Boost confidence for mid-range scores
        confidence[(scores >= 40) & (scores <= 70)] *= 1.1
        
        return np.clip(confidence * 100, 0, 100)
    
    def _determine_trend_direction(self, future_data: pd.DataFrame, recent_data: pd.DataFrame) -> np.ndarray:
        """Determine trend direction for each census tract"""
        
        trends = []
        
        for idx, row in future_data.iterrows():
            census_tract = row['census_tract']
            
            # Find corresponding recent data
            recent_row = recent_data[recent_data['census_tract'] == census_tract]
            
            if not recent_row.empty:
                # Look for trend indicators
                yoy_changes = [col for col in self.feature_columns if '_yoy_change' in col]
                
                if yoy_changes:
                    avg_change = recent_row[yoy_changes].mean().mean()
                    
                    if avg_change > 0.05:
                        trends.append('improving')
                    elif avg_change < -0.05:
                        trends.append('declining')
                    else:
                        trends.append('stable')
                else:
                    trends.append('stable')
            else:
                trends.append('stable')
        
        return np.array(trends)
    
    def create_comprehensive_timeline(self) -> pd.DataFrame:
        """Create comprehensive timeline with historical and predicted data"""
        
        logger.info("Creating comprehensive opportunity score timeline...")
        
        timeline_data = []
        
        # Add historical data
        for year, yearly_scores in self.historical_data.items():
            if hasattr(self, 'yearly_scores') and year in self.yearly_scores:
                hist_data = self.yearly_scores[year][['census_tract', 'opportunity_score']].copy()
                hist_data['year'] = year
                hist_data['data_type'] = 'historical'
                hist_data['score'] = hist_data['opportunity_score']
                timeline_data.append(hist_data[['census_tract', 'year', 'score', 'data_type']])
        
        # Add prediction data
        for year, pred_data in self.predictions.items():
            pred_df = pred_data[['census_tract', 'predicted_opportunity_score', 'prediction_confidence', 'trend_direction']].copy()
            pred_df['year'] = year
            pred_df['data_type'] = 'predicted'
            pred_df['score'] = pred_df['predicted_opportunity_score']
            timeline_data.append(pred_df[['census_tract', 'year', 'score', 'data_type']])
        
        if timeline_data:
            comprehensive_timeline = pd.concat(timeline_data, ignore_index=True)
            comprehensive_timeline = comprehensive_timeline.sort_values(['census_tract', 'year'])
            
            logger.info(f"Created comprehensive timeline with {len(comprehensive_timeline)} records")
            
            return comprehensive_timeline
        else:
            logger.warning("No timeline data available")
            return pd.DataFrame()
    
    def save_models_and_predictions(self, output_dir: str = None):
        """Save trained models and predictions"""
        
        if output_dir is None:
            output_dir = self.config.MODELS_DIR
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        models_file = output_path / f"temporal_opportunity_models_{timestamp}.pkl"
        with open(models_file, 'wb') as f:
            pickle.dump({
                'year_models': self.year_models,
                'feature_columns': self.feature_columns,
                'training_years': self.training_years,
                'prediction_years': self.prediction_years
            }, f)
        
        # Save predictions
        predictions_file = output_path / f"opportunity_predictions_{timestamp}.json"
        
        # Convert predictions to JSON-serializable format
        predictions_json = {}
        for year, pred_df in self.predictions.items():
            predictions_json[str(year)] = pred_df.to_dict('records')
        
        with open(predictions_file, 'w') as f:
            json.dump(predictions_json, f, indent=2)
        
        logger.info(f"Saved models to {models_file}")
        logger.info(f"Saved predictions to {predictions_file}")
        
        return models_file, predictions_file
    
    def run_full_forecast_pipeline(self) -> Dict:
        """Run the complete temporal forecasting pipeline"""
        
        logger.info("Starting complete temporal opportunity forecasting pipeline...")
        
        try:
            # Step 1: Load multi-year data
            yearly_data = self.load_multi_year_data()
            
            if not yearly_data:
                raise ValueError("No yearly data loaded")
            
            # Step 2: Calculate opportunity scores for each year
            yearly_scores = self.calculate_yearly_opportunity_scores()
            self.yearly_scores = yearly_scores
            
            # Step 3: Create temporal features
            temporal_data = self.create_temporal_features(yearly_scores)
            
            # Step 4: Prepare training data
            training_data, feature_columns = self.prepare_training_data(temporal_data)
            
            # Step 5: Train year-specific models
            year_models = self.train_yearly_models(training_data)
            
            # Step 6: Predict future scores
            predictions = self.predict_future_scores(temporal_data)
            
            # Step 7: Create comprehensive timeline
            timeline = self.create_comprehensive_timeline()
            
            # Step 8: Save results
            model_file, pred_file = self.save_models_and_predictions()
            
            # Compile results
            results = {
                'success': True,
                'yearly_data_loaded': len(yearly_data),
                'models_trained': len(year_models),
                'predictions_generated': len(predictions),
                'timeline_records': len(timeline),
                'model_file': str(model_file),
                'predictions_file': str(pred_file),
                'summary': self._generate_summary()
            }
            
            logger.info("Temporal forecasting pipeline completed successfully!")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in forecasting pipeline: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics"""
        
        summary = {
            'training_years': self.training_years,
            'prediction_years': self.prediction_years,
            'models_performance': {},
            'prediction_summary': {}
        }
        
        # Model performance summary
        for year, model_info in self.year_models.items():
            summary['models_performance'][str(year)] = {
                'best_model': model_info['best_model'],
                'r2_score': round(model_info['performance']['r2_score'], 3),
                'mae': round(model_info['performance']['mae'], 3)
            }
        
        # Prediction summary
        for year, pred_data in self.predictions.items():
            summary['prediction_summary'][str(year)] = {
                'census_tracts': len(pred_data),
                'avg_predicted_score': round(pred_data['predicted_opportunity_score'].mean(), 2),
                'score_range': [
                    round(pred_data['predicted_opportunity_score'].min(), 2),
                    round(pred_data['predicted_opportunity_score'].max(), 2)
                ]
            }
        
        return summary


def create_temporal_forecasts(config: LendingConfig = None) -> Dict:
    """Main function to create temporal opportunity forecasts"""
    
    logger.info("Creating temporal opportunity forecasts...")
    
    # Initialize forecaster
    forecaster = TemporalOpportunityForecaster(config)
    
    # Run complete pipeline
    results = forecaster.run_full_forecast_pipeline()
    
    return results


if __name__ == "__main__":
    # Run temporal forecasting
    config = LendingConfig()
    results = create_temporal_forecasts(config)
    
    if results['success']:
        print("✅ Temporal opportunity forecasting completed successfully!")
        print(f"📊 Summary: {results['summary']}")
    else:
        print(f"❌ Forecasting failed: {results['error']}")