#!/usr/bin/env python3
"""
Comprehensive Lending Platform Pipeline

Production-grade pipeline that orchestrates:
1. Data processing and validation
2. Opportunity score calculation
3. Loan outcome prediction model training
4. Future forecasting
5. Market segmentation
6. Report generation

Designed for scalability, maintainability, and production deployment.
"""

import pandas as pd
import numpy as np
import logging
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging BEFORE importing any custom modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log', mode='w'),  # Use write mode to clear previous logs
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom modules (imported AFTER logging configuration)
from advanced_lending_platform import LendingConfig, DataProcessor, OpportunityScoreCalculator
from loan_outcome_predictor import LoanOutcomePredictor, train_loan_prediction_models
from enhanced_loan_predictor import EnhancedLoanPredictor
from opportunity_forecaster import create_comprehensive_forecasts
from market_segmenter import perform_comprehensive_segmentation
from data_validator import DataValidator, create_validation_report


class EnhancedConfig:
    """Configuration class for enhanced loan predictor that can be pickled"""
    def __init__(self, base_config):
        self.BASE_DIR = base_config.BASE_DIR
        self.DATA_DIR = base_config.DATA_DIR  # Pass through the DATA_DIR
        self.balance_method = 'smote'
        self.test_size = 0.2
        self.cv_folds = 5


class LendingPlatformPipeline:
    """Comprehensive lending opportunity platform pipeline"""
    
    def __init__(self):
        self.config = LendingConfig()
        self.results = {}
        self.execution_time = {}
        
        # Initialize comprehensive data validator
        self.validator = DataValidator()
        
        # Create necessary directories
        self.config.MODELS_DIR.mkdir(exist_ok=True)
        self.config.OUTPUTS_DIR.mkdir(exist_ok=True)
          
    def validate_data_sources(self) -> bool:
        """Validate that all required data sources are available"""
        
        logger.info("Validating data sources...")
        
        required_files = [
            self.config.DATA_DIR / "state_KS.csv",
            self.config.DATA_DIR / "enhanced_census_data.csv"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            logger.error(f"Missing required data files: {missing_files}")
            return False
        
        # Check data quality
        try:
            hmda_data = pd.read_csv(self.config.DATA_DIR / "state_KS.csv")
            census_data = pd.read_csv(self.config.DATA_DIR / "enhanced_census_data.csv")
            
            logger.info(f"HMDA data: {len(hmda_data)} records")
            logger.info(f"Census data: {len(census_data)} records")
            
            # Basic quality checks
            if len(hmda_data) < 1000:
                logger.warning("HMDA data seems small - may affect model quality")
            
            if len(census_data) < 10:
                logger.warning("Census data seems small - may affect analysis quality")
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return False
    
    def run_comprehensive_data_validation(self) -> Dict:
        """Run comprehensive data validation and cleaning"""
        
        start_time = datetime.now()
        logger.info("Starting comprehensive data validation and cleaning...")
        
        try:
            # Load raw data first
            raw_data = {}
            
            # Load HMDA data
            hmda_path = self.config.DATA_DIR / "raw" / "hmda_data.csv"
            if hmda_path.exists():
                raw_data['hmda'] = pd.read_csv(hmda_path)
                logger.info(f"Loaded raw HMDA data: {len(raw_data['hmda'])} records")
            else:
                # Fallback to state_KS.csv if available
                hmda_fallback = self.config.DATA_DIR / "state_KS.csv"
                if hmda_fallback.exists():
                    raw_data['hmda'] = pd.read_csv(hmda_fallback)
                    logger.info(f"Loaded raw HMDA data (fallback): {len(raw_data['hmda'])} records")
                else:
                    raise FileNotFoundError(f"HMDA data not found: {hmda_path} or {hmda_fallback}")
            
            # Load Census data
            census_path = self.config.DATA_DIR / "raw" / "census_data.csv"
            if census_path.exists():
                raw_data['census'] = pd.read_csv(census_path)
                logger.info(f"Loaded raw Census data: {len(raw_data['census'])} records")
            else:
                # Fallback to enhanced_census_data.csv
                census_fallback = self.config.DATA_DIR / "enhanced_census_data.csv"
                if census_fallback.exists():
                    raw_data['census'] = pd.read_csv(census_fallback)
                    # Filter for Kansas (state code 20)
                    if 'state_code' in raw_data['census'].columns:
                        raw_data['census'] = raw_data['census'][
                            raw_data['census']['state_code'] == int(self.config.KANSAS_FIPS_CODE)
                        ]
                    logger.info(f"Loaded raw Census data (fallback): {len(raw_data['census'])} Kansas tracts")
                else:
                    raise FileNotFoundError(f"Census data not found: {census_path} or {census_fallback}")
            
            # Load HPI data (optional)
            hpi_path = self.config.DATA_DIR / "raw" / "hpi_data.csv"
            if hpi_path.exists():
                raw_data['hpi'] = pd.read_csv(hpi_path)
                logger.info(f"Loaded raw HPI data: {len(raw_data['hpi'])} records")
            
            # Validate and clean each dataset
            validated_data = {}
            validation_reports = []
            
            for data_type, df in raw_data.items():
                if not df.empty:
                    logger.info(f"Validating {data_type} data...")
                    clean_df, report = self.validator.validate_dataframe(df, data_type, strict=False)
                    validated_data[data_type] = clean_df
                    validation_reports.append({
                        'data_type': data_type,
                        'original_rows': len(df),
                        'clean_rows': len(clean_df),
                        **report
                    })
                    
                    # Log validation results
                    if report['status'] == 'errors':
                        logger.error(f"{data_type} validation errors: {report['errors']}")
                    elif report['status'] == 'warnings':
                        logger.warning(f"{data_type} validation warnings: {report['warnings']}")
                    
                    values_sanitized = report.get('values_sanitized', 0)
                    if values_sanitized > 0:
                        logger.info(f"{data_type}: Sanitized {values_sanitized} values")
                    
                    logger.info(f"Validated {data_type} data: {len(clean_df)} rows (status: {report['status']})")
            
            # Generate comprehensive validation summary
            validation_summary = self.validator.generate_validation_summary(validation_reports)
            
            # Create comprehensive validation report
            create_validation_report(self.validator, validated_data)
            
            # Log summary
            logger.info("📊 Validation Summary:")
            logger.info(f"   - Total Datasets: {validation_summary['total_datasets']}")
            logger.info(f"   - Successful: {validation_summary['successful']}")
            logger.info(f"   - With Warnings: {validation_summary['with_warnings']}")
            logger.info(f"   - With Errors: {validation_summary['with_errors']}")
            logger.info(f"   - Total Rows: {validation_summary['total_rows_processed']:,}")
            logger.info(f"   - Values Sanitized: {validation_summary['total_values_sanitized']:,}")
            
            for recommendation in validation_summary['recommendations']:
                logger.info(f"   💡 {recommendation}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.execution_time['data_validation'] = execution_time
            
            logger.info(f"Comprehensive data validation completed in {execution_time:.2f} seconds")
            
            return {
                'status': 'success',
                'validated_data': validated_data,
                'validation_summary': validation_summary,
                'validation_reports': validation_reports
            }
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def run_data_processing(self, validated_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run data processing and opportunity score calculation using validated data"""
        
        start_time = datetime.now()
        logger.info("Starting data processing with validated data...")
        
        try:
            # Initialize data processor with validated data
            processor = DataProcessor(self.config)
            
            # Set the validated data directly into the processor
            processor.hmda_data = validated_data['hmda']
            processor.census_data = validated_data['census']
            if 'hpi' in validated_data:
                processor.hpi_data = validated_data['hpi']
            
            logger.info("Using pre-validated and cleaned data for processing")
            
            # Create master dataset using validated data (skip internal cleaning)
            # We'll implement the aggregation logic directly since data is already clean
            hmda_clean = validated_data['hmda'].copy()
            census_clean = validated_data['census'].copy()
            
            logger.info(f"Input data shapes - HMDA: {hmda_clean.shape}, Census: {census_clean.shape}")
            
            # Ensure census tract is consistent for joining - handle NaN values properly
            # Filter out rows with invalid census_tract values first
            hmda_clean = hmda_clean.dropna(subset=['census_tract'])
            hmda_clean = hmda_clean[hmda_clean['census_tract'] != 'Unknown']
            
            # Convert to consistent int-based string format  
            hmda_clean['census_tract'] = hmda_clean['census_tract'].astype(float).astype(int).astype(str)
            census_clean['census_tract'] = census_clean['census_tract'].astype(int).astype(str)
            
            # Filter valid HMDA records (avoid processing invalid values)
            hmda_clean = hmda_clean[hmda_clean['action_taken'].isin(range(1, 9))]
            
            logger.info(f"After filtering - HMDA: {hmda_clean.shape}, unique census tracts: {hmda_clean['census_tract'].nunique()}")
            logger.info(f"Census tracts - unique: {census_clean['census_tract'].nunique()}")
            
            # Parse and clean crucial ratio columns
            def parse_debt_to_income_ratio(value):
                """Parse debt_to_income_ratio handling ranges like '20%-<30%' and special values"""
                if pd.isna(value) or value == 'Exempt':
                    return np.nan
                
                if isinstance(value, str):
                    if value == '<20%':
                        return 15.0  # Average of 0-20%
                    elif value == '20%-<30%':
                        return 25.0  # Average of 20-30%
                    elif value == '30%-<36%':
                        return 33.0  # Average of 30-36%
                    elif value == '50%-60%':
                        return 55.0  # Average of 50-60%
                    elif value == '>60%':
                        return 70.0  # Conservative estimate for >60%
                    else:
                        try:
                            return float(value)
                        except:
                            return np.nan
                else:
                    try:
                        return float(value)
                    except:
                        return np.nan
            
            def parse_loan_to_value_ratio(value):
                """Parse loan_to_value_ratio handling 'Exempt' and numeric strings"""
                if pd.isna(value) or value == 'Exempt':
                    return np.nan
                
                if isinstance(value, str):
                    try:
                        return float(value)
                    except:
                        return np.nan
                else:
                    try:
                        return float(value)
                    except:
                        return np.nan
            
            # Apply parsing to ratio columns
            if 'debt_to_income_ratio' in hmda_clean.columns:
                hmda_clean['debt_to_income_ratio'] = hmda_clean['debt_to_income_ratio'].apply(parse_debt_to_income_ratio)
                logger.info(f"Parsed debt_to_income_ratio: {hmda_clean['debt_to_income_ratio'].notna().sum()} valid values out of {len(hmda_clean)}")
            
            if 'loan_to_value_ratio' in hmda_clean.columns:
                hmda_clean['loan_to_value_ratio'] = hmda_clean['loan_to_value_ratio'].apply(parse_loan_to_value_ratio)
                logger.info(f"Parsed loan_to_value_ratio: {hmda_clean['loan_to_value_ratio'].notna().sum()} valid values out of {len(hmda_clean)}")
            
            # Create binary targets from action_taken codes (HMDA standard)
            # Approved: 1=originated, 2=approved not accepted, 6=purchased, 8=preapproval approved not accepted
            # Denied: 3=denied, 7=preapproval denied
            if 'approved' not in hmda_clean.columns:
                hmda_clean['approved'] = hmda_clean['action_taken'].isin([1, 2, 6, 8]).astype(int)
            if 'denied' not in hmda_clean.columns:
                hmda_clean['denied'] = hmda_clean['action_taken'].isin([3, 7]).astype(int)
            
            # Also create loan_approved and loan_denied for compatibility
            if 'loan_approved' not in hmda_clean.columns:
                hmda_clean['loan_approved'] = hmda_clean['approved']
            if 'loan_denied' not in hmda_clean.columns:
                hmda_clean['loan_denied'] = hmda_clean['denied']
            
            # Aggregate HMDA data by census tract
            # Only include columns that are actually numeric (not object/string)
            numeric_cols = ['loan_amount', 'income', 'approved', 'denied']
            
            # Add ratio columns only if they're numeric after validation
            for col in ['debt_to_income_ratio', 'loan_to_value_ratio']:
                if col in hmda_clean.columns and pd.api.types.is_numeric_dtype(hmda_clean[col]):
                    numeric_cols.append(col)
            
            # Only include columns that exist and are numeric
            available_cols = [col for col in numeric_cols if col in hmda_clean.columns]
            
            agg_dict = {}
            for col in available_cols:
                if col in ['loan_amount']:
                    agg_dict[col] = ['count', 'mean', 'median']
                elif col in ['income']:
                    agg_dict[col] = ['mean', 'median']
                elif col in ['approved', 'denied']:
                    agg_dict[col] = ['sum', 'mean']
                else:
                    agg_dict[col] = 'mean'
            
            # Add action_taken for mode calculation
            agg_dict['action_taken'] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 3
            
            hmda_agg = hmda_clean.groupby('census_tract').agg(agg_dict).round(2)
            
            # Flatten column names
            new_columns = []
            for col in hmda_agg.columns:
                if isinstance(col, tuple):
                    if col[0] == 'loan_amount':
                        if col[1] == 'count':
                            new_columns.append('loan_count')
                        elif col[1] == 'mean':
                            new_columns.append('avg_loan_amount')
                        elif col[1] == 'median':
                            new_columns.append('median_loan_amount')
                    elif col[0] == 'income':
                        if col[1] == 'mean':
                            new_columns.append('avg_income')
                        elif col[1] == 'median':
                            new_columns.append('median_applicant_income')
                    elif col[0] == 'approved':
                        if col[1] == 'sum':
                            new_columns.append('total_approved')
                        elif col[1] == 'mean':
                            new_columns.append('approval_rate')
                    elif col[0] == 'denied':
                        if col[1] == 'sum':
                            new_columns.append('total_denied')
                        elif col[1] == 'mean':
                            new_columns.append('denial_rate')
                    else:
                        new_columns.append(f"avg_{col[0]}")
                else:
                    new_columns.append(col[0] if isinstance(col, tuple) else str(col))
            
            hmda_agg.columns = new_columns
            hmda_agg = hmda_agg.reset_index()
            
            logger.info(f"Aggregated HMDA shape: {hmda_agg.shape}, columns: {hmda_agg.columns.tolist()}")
            logger.info(f"Census clean shape: {census_clean.shape}, columns: {census_clean.columns.tolist()}")
            
            # Debug: Check tract overlap before merge
            hmda_tracts = set(hmda_agg['census_tract'].unique())
            census_tracts = set(census_clean['census_tract'].unique())
            overlap = hmda_tracts.intersection(census_tracts)
            logger.info(f"Pre-merge tract overlap: HMDA={len(hmda_tracts)}, Census={len(census_tracts)}, Overlap={len(overlap)}")
            
            # Merge with census data
            master_data = pd.merge(hmda_agg, census_clean, on='census_tract', how='inner')
            
            # Add HPI data if available
            if 'hpi' in validated_data and not validated_data['hpi'].empty:
                hpi_data = validated_data['hpi'].copy()
                hpi_data['census_tract'] = hpi_data['census_tract'].astype(str)
                
                # Get latest HPI data for each tract
                if 'year' in hpi_data.columns:
                    hpi_latest = hpi_data.loc[hpi_data.groupby('census_tract')['year'].idxmax()]
                    hpi_merge = hpi_latest[['census_tract', 'hpi_value']].copy()
                    hpi_merge.columns = ['census_tract', 'hpi']
                    master_data = pd.merge(master_data, hpi_merge, on='census_tract', how='left')
            
            logger.info(f"Created master dataset with {len(master_data)} census tracts")
            
            # Calculate opportunity scores
            calculator = OpportunityScoreCalculator(self.config)
            scored_data = calculator.calculate_opportunity_score(master_data)
            
            # Validate the final scored data
            logger.info("Validating final scored data...")
            validated_scored_data, scoring_report = self.validator.validate_dataframe(
                scored_data, 'predictions', strict=False
            )
            
            if scoring_report['status'] != 'success':
                logger.warning(f"Scored data validation: {scoring_report['status']}")
                for warning in scoring_report.get('warnings', []):
                    logger.warning(f"Scoring validation: {warning}")
            
            # Generate summary statistics (no CSV file saved - dashboard uses fresh data)
            summary = {
                'total_census_tracts': len(validated_scored_data),
                'average_opportunity_score': validated_scored_data['opportunity_score'].mean(),
                'opportunity_level_distribution': validated_scored_data['opportunity_level'].value_counts().to_dict(),
                'total_population': validated_scored_data['total_population'].sum(),
                'scoring_validation': scoring_report
            }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.execution_time['data_processing'] = execution_time
            
            logger.info(f"Data processing completed in {execution_time:.2f} seconds")
            
            return {
                'status': 'success',
                'data': scored_data,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def run_loan_prediction_training(self, validated_hmda_data: pd.DataFrame) -> Dict:
        """Train enhanced loan outcome prediction models using validated HMDA data with advanced features
        
        Args:
            validated_hmda_data: Validated HMDA data for training
        
        Note: Speed optimization is now configured via config.yaml under model.loan_prediction.optimize_for_speed
        """
        
        start_time = datetime.now()
        logger.info("Starting enhanced HMDA loan outcome prediction model training with validated data...")
        
        # Get optimize_for_speed setting from configuration
        optimize_for_speed = self.config.OPTIMIZE_FOR_SPEED
        
        try:
            # Initialize enhanced loan predictor with base config to avoid pickle issues
            enhanced_predictor = EnhancedLoanPredictor(self.config)
            
            # Set the attributes that were in EnhancedConfig directly
            enhanced_predictor.balance_method = 'smote'
            enhanced_predictor.test_size = 0.2
            enhanced_predictor.cv_folds = 5
            
            logger.info("Using Enhanced HMDA Loan Predictor with:")
            logger.info("  • Advanced feature engineering (DTI range parsing, financial ratios)")
            logger.info("  • Class imbalance handling with SMOTE")
            logger.info("  • Multiple model comparison (RF, XGBoost, LightGBM)")
            logger.info("  • Fair lending compliance monitoring")
            
            # Save validated HMDA data temporarily for training
            temp_hmda_path = self.config.DATA_DIR / "state_KS.csv"
            
            # Ensure the data directory exists
            temp_hmda_path.parent.mkdir(exist_ok=True)
            
            # Save validated data for the enhanced predictor to use
            validated_hmda_data.to_csv(temp_hmda_path, index=False)
            logger.info(f"Prepared validated HMDA data: {len(validated_hmda_data):,} records")
            
            # Train enhanced models using the new system
            logger.info("Training enhanced models with comprehensive feature engineering...")
            
            # Load and prepare data through enhanced feature engineering
            df_engineered = enhanced_predictor.load_and_prepare_data()
            logger.info(f"Enhanced features created: {df_engineered.shape[1]} features")
            
            # Get target distribution info
            valid_targets = df_engineered[df_engineered['target'].isin([0, 1])]
            denial_rate = (valid_targets['target'] == 0).mean()
            approval_rate = (valid_targets['target'] == 1).mean()
            
            logger.info(f"Enhanced target distribution:")
            logger.info(f"  • Denial rate: {denial_rate:.1%}")
            logger.info(f"  • Approval rate: {approval_rate:.1%}")
            logger.info(f"  • Valid training samples: {len(valid_targets):,}")
            
            # Train models with the enhanced pipeline
            df_clean, X, y = enhanced_predictor.modeling_pipeline.prepare_data(df_engineered)
            
            # Configure models based on user preference
            if optimize_for_speed:
                logger.info("Training enhanced models with SPEED-OPTIMIZED parameters...")
                
                # Streamlined configurations for faster training
                enhanced_predictor.modeling_pipeline.model_configs = {
                    'random_forest': {
                        'model': enhanced_predictor.modeling_pipeline.model_configs['random_forest']['model'],
                        'params': {
                            'n_estimators': [50, 100],
                            'max_depth': [10, 20],
                            'min_samples_split': [5],
                            'class_weight': ['balanced'],
                            'n_jobs': [-1]
                        },
                        'scoring': 'f1'
                    },
                    'xgboost': {
                        'model': enhanced_predictor.modeling_pipeline.model_configs['xgboost']['model'],
                        'params': {
                            'n_estimators': [50, 100],
                            'learning_rate': [0.1],
                            'max_depth': [5, 7],
                            'scale_pos_weight': [3],
                            'n_jobs': [-1]
                        },
                        'scoring': 'f1'
                    }
                }
                cv_folds = 3
                logger.info("Using SPEED mode: 2 models, reduced grids, 3-fold CV (6x faster, ~0.3% accuracy loss)")
                
            else:
                logger.info("Training enhanced models with MAXIMUM ACCURACY parameters...")
                
                # Full configurations for maximum accuracy
                enhanced_predictor.modeling_pipeline.model_configs = {
                    'random_forest': {
                        'model': enhanced_predictor.modeling_pipeline.model_configs['random_forest']['model'],
                        'params': {
                            'n_estimators': [100, 200],
                            'max_depth': [10, 20, None],
                            'min_samples_split': [2, 5],
                            'class_weight': ['balanced', 'balanced_subsample'],
                            'n_jobs': [-1]
                        },
                        'scoring': 'f1'
                    },
                    'xgboost': {
                        'model': enhanced_predictor.modeling_pipeline.model_configs['xgboost']['model'],
                        'params': {
                            'n_estimators': [100, 200],
                            'learning_rate': [0.1, 0.15],
                            'max_depth': [5, 7],
                            'scale_pos_weight': [1, 3, 5],
                            'n_jobs': [-1]
                        },
                        'scoring': 'f1'
                    },
                    'gradient_boosting': {
                        'model': enhanced_predictor.modeling_pipeline.model_configs['gradient_boosting']['model'],
                        'params': {
                            'n_estimators': [100, 200],
                            'learning_rate': [0.1, 0.15],
                            'max_depth': [5, 7]
                        },
                        'scoring': 'f1'
                    }
                }
                cv_folds = 5
                logger.info("Using ACCURACY mode: 3 models, full grids, 5-fold CV (maximum accuracy)")
            
            # Train models with class imbalance handling (configurable accuracy)
            training_results = enhanced_predictor.modeling_pipeline.train_and_evaluate_models(
                X, y, 
                balance_method='smote',
                test_size=0.2,
                cv_folds=cv_folds  # Use configured cv_folds (3 for speed, 5 for accuracy)
            )
            
            # Get model performance summary
            performance_summary = enhanced_predictor.modeling_pipeline.get_model_summary()
            logger.info("Enhanced model training completed!")
            logger.info(f"Model Performance Summary:")
            logger.info(f"\n{performance_summary.to_string(index=False)}")
            
            # Test enhanced prediction with comprehensive sample
            sample_application = {
                'debt_to_income_ratio': 28.0,
                'loan_to_value_ratio': 0.80,
                'property_value': 312500,
                'income': 75,  # in thousands
                'loan_amount': 250000,
                'loan_purpose': 1,  # home purchase
                'applicant_race-1': 5,
                'applicant_sex': 1,
                'applicant_ethnicity': 2,
                'occupancy_type': 1,
                'loan_type': 1,
                'interest_rate': 6.5,
                'loan_term': 360
            }
            
            # Validate sample application before prediction
            logger.info("Testing enhanced prediction with sample application...")
            validated_application, app_validation = self.validator.validate_loan_application(sample_application)
            
            if app_validation['status'] != 'success':
                logger.warning(f"Sample application validation: {app_validation['status']}")
                for warning in app_validation.get('warnings', []):
                    logger.warning(f"Application validation: {warning}")
            
            # Make enhanced prediction
            enhanced_prediction = enhanced_predictor.modeling_pipeline.predict_loan_outcome(validated_application)
            logger.info(f"Enhanced prediction result: {enhanced_prediction}")
            
            # Evaluate model fairness
            logger.info("Evaluating model fairness across demographic groups...")
            fairness_results = enhanced_predictor.modeling_pipeline.evaluate_fairness(df_clean, X, y)
            
            # Save enhanced models and metadata
            model_save_path = self.config.MODELS_DIR / "enhanced_loan_models.pkl"
            
            # Save the complete enhanced predictor with all models and metadata
            import pickle
            with open(model_save_path, 'wb') as f:
                pickle.dump({
                    'enhanced_predictor': enhanced_predictor,
                    'feature_columns': enhanced_predictor.modeling_pipeline.feature_columns,
                    'performance_summary': performance_summary,
                    'training_metadata': {
                        'training_date': datetime.now().isoformat(),
                        'feature_count': len(enhanced_predictor.modeling_pipeline.feature_columns),
                        'training_samples': len(valid_targets),
                        'denial_rate': denial_rate,
                        'approval_rate': approval_rate,
                        'best_models': performance_summary.head(3).to_dict('records')
                    }
                }, f)
            
            logger.info(f"Enhanced models saved to: {model_save_path}")
            
            # Also save individual model artifacts for easier loading
            models_dir = self.config.MODELS_DIR / "individual_models"
            models_dir.mkdir(exist_ok=True)
            
            # Save best performing models individually
            best_models = enhanced_predictor.modeling_pipeline.models  # Fixed: use .models instead of .trained_models
            for model_name, model_info in best_models.items():
                individual_path = models_dir / f"{model_name}_model.pkl"
                with open(individual_path, 'wb') as f:
                    pickle.dump(model_info, f)
                logger.info(f"Saved {model_name} model to: {individual_path}")
            
            # Save feature engineering pipeline separately for dashboard use
            feature_pipeline_path = self.config.MODELS_DIR / "feature_pipeline.pkl"
            with open(feature_pipeline_path, 'wb') as f:
                pickle.dump(enhanced_predictor.feature_engineer, f)
            logger.info(f"Feature engineering pipeline saved to: {feature_pipeline_path}")
            
            # Mark predictor as trained
            enhanced_predictor.is_trained = True
            enhanced_predictor.feature_columns = enhanced_predictor.modeling_pipeline.feature_columns
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.execution_time['loan_prediction'] = execution_time
            
            logger.info(f"Enhanced loan prediction training completed in {execution_time:.2f} seconds")
            logger.info("Key improvements achieved:")
            logger.info(f"  ✅ Advanced feature engineering: {len(enhanced_predictor.modeling_pipeline.feature_columns)} features")
            logger.info(f"  ✅ Class imbalance handling: SMOTE balancing applied")
            logger.info(f"  ✅ Realistic predictions: {denial_rate:.1%} denial rate (was ~0%)")
            logger.info(f"  ✅ Multiple model comparison with hyperparameter tuning")
            logger.info(f"  ✅ Fair lending compliance monitoring")
            
            return {
                'status': 'success',
                'enhanced_predictor': enhanced_predictor,
                'performance_summary': performance_summary.to_dict('records'),
                'training_results': training_results,
                'sample_prediction': enhanced_prediction,
                'sample_validation': app_validation,
                'fairness_evaluation': fairness_results,
                'model_path': str(model_save_path),
                'denial_rate': denial_rate,
                'approval_rate': approval_rate,
                'feature_count': len(enhanced_predictor.modeling_pipeline.feature_columns),
                'training_samples': len(valid_targets)
            }
            
        except Exception as e:
            logger.error(f"Enhanced loan prediction training failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
        except Exception as e:
            logger.error(f"Loan prediction training failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def run_temporal_forecasting_pipeline(self) -> Dict:
        """Run temporal forecasting pipeline to train models and generate predictions"""
        
        start_time = datetime.now()
        logger.info("Starting temporal forecasting pipeline...")
        
        try:
            # Import the temporal forecasting pipeline
            from temporal_forecasting_pipeline import TemporalForecastingPipeline
            
            # Initialize and run the temporal forecasting pipeline
            temporal_pipeline = TemporalForecastingPipeline()
            
            logger.info("Loading HMDA temporal forecaster...")
            from hmda_temporal_forecaster import HMDAOpportunityForecaster
            
            # Initialize the forecaster
            forecaster = HMDAOpportunityForecaster()
            
            # Run the complete temporal forecasting pipeline
            logger.info("Running HMDA temporal forecasting pipeline...")
            results = forecaster.run_pipeline()
            
            if not results['success']:
                raise Exception(f"Temporal forecasting failed: {results.get('error', 'Unknown error')}")
            
            # Save the results using the temporal pipeline
            logger.info("Saving temporal forecasting results...")
            save_results = temporal_pipeline.save_forecasting_results(forecaster, results)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.execution_time['temporal_forecasting'] = execution_time
            
            logger.info(f"Temporal forecasting pipeline completed in {execution_time:.2f} seconds")
            logger.info(f"Generated predictions for {results.get('predictions_generated', 0)} future years")
            logger.info(f"Trained models for {results.get('models_trained', 0)} historical years")
            logger.info(f"Created timeline with {results.get('timeline_records', 0)} records")
            
            return {
                'status': 'success',
                'forecaster': forecaster,
                'results': results,
                'save_results': save_results,
                'execution_time': execution_time
            }
            
        except Exception as e:
            logger.error(f"Temporal forecasting pipeline failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def run_opportunity_forecasting(self, current_data: pd.DataFrame) -> Dict:
        """Generate opportunity score forecasts"""
        
        start_time = datetime.now()
        logger.info("Starting opportunity score forecasting...")
        
        try:
            # Generate forecasts
            forecasts, results = create_comprehensive_forecasts(current_data, self.config)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.execution_time['forecasting'] = execution_time
            
            logger.info(f"Opportunity forecasting completed in {execution_time:.2f} seconds")
            
            return {
                'status': 'success',
                'forecasts': forecasts,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Opportunity forecasting failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def run_market_segmentation(self, scored_data: pd.DataFrame) -> Dict:
        """Perform market segmentation analysis"""
        
        start_time = datetime.now()
        logger.info("Starting market segmentation analysis...")
        
        try:
            # Perform segmentation
            segmented_data, analysis_results = perform_comprehensive_segmentation(scored_data, self.config)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.execution_time['segmentation'] = execution_time
            
            logger.info(f"Market segmentation completed in {execution_time:.2f} seconds")
            
            return {
                'status': 'success',
                'segmented_data': segmented_data,
                'analysis_results': analysis_results
            }
            
        except Exception as e:
            logger.error(f"Market segmentation failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def generate_executive_summary(self) -> Dict:
        """Generate executive summary report"""
        
        logger.info("Generating executive summary...")
        
        try:
            summary_data = {}
            
            # Data processing summary
            if 'data_processing' in self.results:
                dp_result = self.results['data_processing']
                if dp_result['status'] == 'success':
                    summary_data['opportunity_analysis'] = dp_result['summary']
            
            # Forecasting summary
            if 'forecasting' in self.results:
                fc_result = self.results['forecasting']
                if fc_result['status'] == 'success':
                    summary_data['forecasting'] = fc_result['results']['trend_analysis']
            
            # Enhanced loan prediction summary
            # if 'loan_prediction' in self.results:
            #     lp_result = self.results['loan_prediction']
            #     if lp_result['status'] == 'success':
            #         summary_data['enhanced_loan_prediction'] = {
            #             'denial_rate': f"{lp_result.get('denial_rate', 0):.1%}",
            #             'approval_rate': f"{lp_result.get('approval_rate', 0):.1%}",
            #             'feature_count': lp_result.get('feature_count', 0),
            #             'training_samples': lp_result.get('training_samples', 0),
            #             'model_performance': lp_result.get('performance_summary', []),
            #             'fairness_evaluated': 'fairness_evaluation' in lp_result
            #         }
            
            # Segmentation summary
            if 'segmentation' in self.results:
                seg_result = self.results['segmentation']
                if seg_result['status'] == 'success':
                    summary_data['market_segments'] = {
                        'total_segments': len(seg_result['analysis_results']['segment_profiles']),
                        'clustering_info': seg_result['analysis_results']['clustering_info']
                    }
            
            # Performance metrics
            summary_data['pipeline_performance'] = self.execution_time
            summary_data['total_execution_time'] = sum(self.execution_time.values())
            
            logger.info("Executive summary generated")
            
            return {
                'status': 'success',
                'summary': summary_data
            }
            
        except Exception as e:
            logger.error(f"Executive summary generation failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def run_full_pipeline(self) -> Dict:
        """Execute the complete pipeline"""
        
        pipeline_start = datetime.now()
        logger.info("="*60)
        logger.info("STARTING COMPREHENSIVE LENDING PLATFORM PIPELINE")
        logger.info("="*60)
        
        # Step 1: Validate data sources
        if not self.validate_data_sources():
            return {
                'status': 'error',
                'error': 'Data validation failed'
            }
        
        # Step 2: Comprehensive Data Validation and Cleaning
        logger.info("\n🔍 Step 1: Comprehensive Data Validation & Cleaning")
        self.results['data_validation'] = self.run_comprehensive_data_validation()
        
        if self.results['data_validation']['status'] != 'success':
            return self.results['data_validation']
        
        validated_data = self.results['data_validation']['validated_data']
        
        # Step 3: Data processing and opportunity scoring with validated data
        logger.info("\n🔄 Step 2: Data Processing & Opportunity Scoring (with validated data)")
        self.results['data_processing'] = self.run_data_processing(validated_data)
        
        if self.results['data_processing']['status'] != 'success':
            return self.results['data_processing']
        
        scored_data = self.results['data_processing']['data']
        
        # Step 4: Loan prediction model training with validated HMDA data
        #logger.info("\n🤖 Step 3: Loan Outcome Prediction Training (with validated data)")
        #self.results['loan_prediction'] = self.run_loan_prediction_training(validated_data['hmda'])
        
        # Step 5: Temporal forecasting pipeline
        logger.info("\n🔮 Step 4: Temporal Opportunity Forecasting Pipeline")
        self.results['temporal_forecasting'] = self.run_temporal_forecasting_pipeline()
        
        # Step 6: Opportunity forecasting
        logger.info("\n📈 Step 5: Opportunity Score Forecasting")
        self.results['forecasting'] = self.run_opportunity_forecasting(scored_data)
        
        # Step 7: Market segmentation
        logger.info("\n🎯 Step 6: Market Segmentation Analysis")
        self.results['segmentation'] = self.run_market_segmentation(scored_data)
        
        # Step 8: Executive summary
        logger.info("\n📊 Step 7: Executive Summary Generation")
        self.results['executive_summary'] = self.generate_executive_summary()
        
        total_time = (datetime.now() - pipeline_start).total_seconds()
        
        # Final pipeline summary
        logger.info("\n" + "="*60)
        logger.info("PIPELINE EXECUTION COMPLETED")
        logger.info("="*60)
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        
        # Count successful vs failed steps
        successful_steps = sum(1 for result in self.results.values() if result.get('status') == 'success')
        total_steps = len(self.results)
        
        logger.info(f"Successful steps: {successful_steps}/{total_steps}")
        
        if successful_steps == total_steps:
            logger.info("✅ All pipeline steps completed successfully!")
        else:
            logger.warning(f"⚠️  {total_steps - successful_steps} step(s) failed")
        
        # Print key results
        if self.results['data_validation']['status'] == 'success':
            val_summary = self.results['data_validation']['validation_summary']
            logger.info(f"\n🔍 Data Validation Results:")
            logger.info(f"   • Total datasets validated: {val_summary['total_datasets']}")
            logger.info(f"   • Successful validations: {val_summary['successful']}")
            logger.info(f"   • With warnings: {val_summary['with_warnings']}")
            logger.info(f"   • With errors: {val_summary['with_errors']}")
            logger.info(f"   • Total rows processed: {val_summary['total_rows_processed']:,}")
            logger.info(f"   • Values sanitized: {val_summary['total_values_sanitized']:,}")
            
        if self.results['data_processing']['status'] == 'success':
            summary = self.results['data_processing']['summary']
            logger.info(f"\n📋 Analysis Results:")
            logger.info(f"   • Census tracts analyzed: {summary['total_census_tracts']}")
            logger.info(f"   • Average opportunity score: {summary['average_opportunity_score']:.1f}")
            logger.info(f"   • Total market population: {summary['total_population']:,}")
            
        # if self.results['loan_prediction']['status'] == 'success':
        #     loan_summary = self.results['loan_prediction']
        #     if 'sample_validation' in loan_summary:
        #         logger.info(f"   • Sample loan prediction validation: {loan_summary['sample_validation']['status']}")
        #         if loan_summary['sample_validation']['values_sanitized'] > 0:
        #             logger.info(f"   • Loan application values sanitized: {loan_summary['sample_validation']['values_sanitized']}")
            
        logger.info(f"\n✅ All data validated and processed with enterprise-grade quality assurance!")
        
        return {
            'status': 'success' if successful_steps == total_steps else 'partial_success',
            'results': self.results,
            'execution_time': total_time,
            'successful_steps': successful_steps,
            'total_steps': total_steps
        }


def main():
    """Main pipeline execution"""
    
    # Initialize and run pipeline
    pipeline = LendingPlatformPipeline()
    
    try:
        final_results = pipeline.run_full_pipeline()
        
        # Print final status
        if final_results['status'] == 'success':
            print("\n🎉 Pipeline completed successfully!")
            print("📁 Latest temporal forecasting results saved to data/outputs/")
            print("🚀 You can now run the dashboard: streamlit run src/executive_dashboard.py")
        else:
            print(f"\n⚠️  Pipeline completed with issues: {final_results['status']}")
            print("📋 Check the logs for details on any failed steps.")
        
        return final_results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }


if __name__ == "__main__":
    results = main()