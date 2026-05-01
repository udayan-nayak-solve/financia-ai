#!/usr/bin/env python3
"""
Temporal Forecasting Pipeline

Pre-computes temporal forecasting results and saves them for dashboard display.
This runs the HMDA temporal forecasting once and stores results for fast dashboard loading.
"""

import pandas as pd
import numpy as np
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import warnings
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

from hmda_temporal_forecaster import HMDAOpportunityForecaster

# Configuration
# Note: Logging is configured in the main pipeline module
logger = logging.getLogger(__name__)


class TemporalForecastingPipeline:
    """Pipeline for pre-computing and storing temporal forecasting results"""
    
    def __init__(self, data_dir: str = "data/actual", output_dir: str = "data/outputs"):
        """Initialize the pipeline"""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Only latest files (used by dashboard)
        self.latest_results_file = self.output_dir / "temporal_forecasting_results_latest.json"
        self.latest_historical_file = self.output_dir / "historical_opportunity_scores_latest.json"
        self.latest_predictions_file = self.output_dir / "future_predictions_latest.json"
        self.latest_performance_file = self.output_dir / "model_performance_latest.json"
        
        logger.info(f"Pipeline initialized - output directory: {self.output_dir}")
    
    def run_forecasting_pipeline(self) -> Dict[str, Any]:
        """Run the complete temporal forecasting pipeline"""
        
        logger.info("🚀 Starting temporal forecasting pipeline...")
        
        try:
            # Initialize forecaster
            forecaster = HMDAOpportunityForecaster(str(self.data_dir))
            
            # Run forecasting
            results = forecaster.run_full_forecast_pipeline()
            
            if not results['success']:
                raise Exception(f"Forecasting failed: {results.get('error', 'Unknown error')}")
            
            # Extract and save results
            self.save_forecasting_results(forecaster, results)
            
            logger.info("✅ Temporal forecasting pipeline completed successfully!")
            
            return {
                'success': True,
                'message': 'Temporal forecasting pipeline completed successfully',
                'files_created': {
                    'results': str(self.latest_results_file),
                    'historical': str(self.latest_historical_file),
                    'predictions': str(self.latest_predictions_file),
                    'performance': str(self.latest_performance_file)
                },
                'summary': results['summary']
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_forecasting_results(self, forecaster, results: Dict[str, Any]) -> None:
        """Save forecasting results to latest JSON files only"""
        
        logger.info("💾 Saving forecasting results...")
        
        # 1. Save main results summary
        main_results = {
            'success': results['success'],
            'timestamp': datetime.now().isoformat(),
            'yearly_data_loaded': results['yearly_data_loaded'],
            'models_trained': results['models_trained'],
            'predictions_generated': results['predictions_generated'],
            'timeline_records': results['timeline_records'],
            'summary': results['summary']
        }
        
        self._save_json(main_results, self.latest_results_file)
        
        # 2. Save historical opportunity scores by year
        historical_data = {}
        for year, scores_df in forecaster.yearly_scores.items():
            # Convert DataFrame to JSON-serializable format
            historical_data[str(year)] = {
                'census_tracts': len(scores_df),
                'average_score': float(scores_df['opportunity_score'].mean()),
                'score_range': [
                    float(scores_df['opportunity_score'].min()),
                    float(scores_df['opportunity_score'].max())
                ],
                'data': scores_df.to_dict('records')
            }
        
        historical_results = {
            'timestamp': datetime.now().isoformat(),
            'years': list(forecaster.yearly_scores.keys()),
            'total_tracts': sum(len(df) for df in forecaster.yearly_scores.values()),
            'data_by_year': historical_data
        }
        
        self._save_json(historical_results, self.latest_historical_file)
        
        # 3. Save future predictions
        predictions_data = {}
        for year, pred_df in forecaster.predictions.items():
            predictions_data[str(year)] = {
                'census_tracts': len(pred_df),
                'average_predicted_score': float(pred_df['predicted_opportunity_score'].mean()),
                'score_range': [
                    float(pred_df['predicted_opportunity_score'].min()),
                    float(pred_df['predicted_opportunity_score'].max())
                ],
                'high_opportunity_tracts': len(pred_df[pred_df['predicted_opportunity_score'] >= 70]),
                'data': pred_df.to_dict('records')
            }
        
        predictions_results = {
            'timestamp': datetime.now().isoformat(),
            'prediction_years': list(forecaster.predictions.keys()),
            'data_by_year': predictions_data
        }
        
        self._save_json(predictions_results, self.latest_predictions_file)
        
        # 4. Save model performance details
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'training_years': list(forecaster.year_models.keys()),
            'feature_count': len(forecaster.feature_columns),
            'models_by_year': {}
        }
        
        for year, model_info in forecaster.year_models.items():
            performance_data['models_by_year'][str(year)] = {
                'best_model': model_info['best_model'],
                'r2_score': float(model_info['performance']['r2_score']),
                'mse': float(model_info['performance']['mse']),
                'mae': float(model_info['performance']['mae']),
                'model_details': {
                    name: {
                        'r2_score': float(details['r2_score']),
                        'mse': float(details['mse']),
                        'mae': float(details['mae'])
                    }
                    for name, details in model_info['models'].items()
                    if isinstance(details, dict) and 'r2_score' in details
                }
            }
        
        self._save_json(performance_data, self.latest_performance_file)
        
        logger.info(f"📊 Saved results for {len(forecaster.yearly_scores)} historical years")
        logger.info(f"🔮 Saved predictions for {len(forecaster.predictions)} future years")
        logger.info(f"🎯 Saved performance data for {len(forecaster.year_models)} models")
    
    def _save_json(self, data: Dict[str, Any], filepath: Path) -> None:
        """Save data to JSON file with error handling"""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Saved: {filepath}")
        except Exception as e:
            logger.error(f"Error saving {filepath}: {str(e)}")
    
    def create_timeline_analysis(self, forecaster) -> Dict[str, Any]:
        """Create timeline analysis data for dashboard"""
        
        timeline_data = []
        
        # Add historical data
        for year, scores_df in forecaster.yearly_scores.items():
            for _, row in scores_df.iterrows():
                timeline_data.append({
                    'census_tract': str(row['census_tract']),
                    'year': int(year),
                    'opportunity_score': float(row['opportunity_score']),
                    'data_type': 'Historical',
                    'lending_activity': float(row.get('lending_activity', 0)),
                    'approval_rate': float(row.get('approval_rate', 0))
                })
        
        # Add prediction data
        for year, pred_df in forecaster.predictions.items():
            for _, row in pred_df.iterrows():
                timeline_data.append({
                    'census_tract': str(row['census_tract']),
                    'year': int(year),
                    'opportunity_score': float(row['predicted_opportunity_score']),
                    'data_type': 'Predicted',
                    'prediction_confidence': float(row.get('prediction_confidence', 0)),
                    'trend_direction': str(row.get('trend_direction', 'stable'))
                })
        
        return {
            'timeline_records': len(timeline_data),
            'years_covered': sorted(list(set([item['year'] for item in timeline_data]))),
            'census_tracts': len(set([item['census_tract'] for item in timeline_data])),
            'data': timeline_data
        }


def run_temporal_forecasting_pipeline(data_dir: str = "data/actual", output_dir: str = "data/outputs") -> Dict[str, Any]:
    """Main function to run the temporal forecasting pipeline"""
    
    logger.info("🚀 Running Temporal Forecasting Pipeline")
    logger.info("=" * 60)
    
    pipeline = TemporalForecastingPipeline(data_dir, output_dir)
    results = pipeline.run_forecasting_pipeline()
    
    if results['success']:
        logger.info("\n✅ Pipeline completed successfully!")
        logger.info(f"📁 Results saved to: {output_dir}")
        logger.info(f"📊 Summary: {results['summary']}")
    else:
        logger.error(f"\n❌ Pipeline failed: {results['error']}")
    
    return results


if __name__ == "__main__":
    results = run_temporal_forecasting_pipeline()
    
    if results['success']:
        print("\n🎉 Temporal forecasting pipeline completed!")
        print("📈 Results are ready for dashboard display")
        print("\n🚀 Next: Launch dashboard with pre-computed results")
        print("   streamlit run src/executive_dashboard.py")
    else:
        print(f"\n❌ Pipeline failed: {results['error']}")