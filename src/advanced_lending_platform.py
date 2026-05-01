#!/usr/bin/env python3
"""
Advanced Lending Opportunity Platform - Production Grade AI System

A comprehensive lending opportunity analysis platform that:
1. Calculates meaningful opportunity scores based on real market data
2. Predicts loan outcomes (approval/denial) with reasons
3. Forecasts future opportunity scores by census tract
4. Performs market segmentation and clustering
5. Provides executive dashboard capabilities

Author: AI Assistant
Date: October 2025
Version: 2.0
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
import yaml
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')

# Configuration
# Note: Logging is configured in the main pipeline module
logger = logging.getLogger(__name__)


class LendingConfig:
    """Configuration class for the lending platform with YAML support"""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration from YAML file or use defaults"""
        
        # Set base paths
        self.BASE_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.BASE_DIR / "data" / "actual"
        self.MODELS_DIR = self.BASE_DIR / "data" / "models"
        self.OUTPUTS_DIR = self.BASE_DIR / "data" / "outputs"
        
        # Load YAML configuration if provided
        self.config_data = {}
        if config_path is None:
            config_path = self.BASE_DIR / "config" / "config.yaml"
        
        if Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    self.config_data = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                self.config_data = {}
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
        
        # Initialize configuration values
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration values from YAML or set defaults"""
        
        # Model training configuration
        model_config = self.config_data.get('model', {})
        loan_prediction_config = model_config.get('loan_prediction', {})
        
        # Get optimize_for_speed setting from config.yaml (default: False for maximum accuracy)
        self.OPTIMIZE_FOR_SPEED = loan_prediction_config.get('optimize_for_speed', False)
        
        # Log the configuration choice
        if self.OPTIMIZE_FOR_SPEED:
            logger.info("🚀 Speed optimization ENABLED: Using faster training with minimal accuracy loss")
        else:
            logger.info("🎯 Maximum accuracy mode ENABLED: Using full training for best results")
        
        # Kansas state identifiers
        self.KANSAS_STATE_CODE = "KS"
        self.KANSAS_FIPS_CODE = "20"
        
        # HMDA Action Taken codes
        self.ACTION_TAKEN_MAPPING = {
            1: "Loan originated",
            2: "Application approved but not accepted", 
            3: "Application denied",
            4: "Application withdrawn by applicant",
            5: "File closed for incompleteness",
            6: "Purchased loan",
            7: "Preapproval request denied",
            8: "Preapproval request approved but not accepted"
        }
        
        # Denial reasons mapping
        self.DENIAL_REASONS = {
            1: "Debt-to-income ratio",
            2: "Employment history",
            3: "Credit history", 
            4: "Collateral",
            5: "Insufficient cash",
            6: "Unverifiable information",
            7: "Credit application incomplete",
            8: "Mortgage insurance denied",
            9: "Other",
            10: "NA"
        }
        
        # Opportunity Score Parameters (Configurable)
        self.OPPORTUNITY_WEIGHTS = {
            'market_accessibility': 0.30,    # 30% weight
            'risk_factors': 0.25,            # 25% weight  
            'economic_indicators': 0.25,     # 25% weight
            'lending_activity': 0.20         # 20% weight
        }
        
        # Risk tolerance levels
        self.RISK_TOLERANCE = {
            'conservative': 0.3,
            'moderate': 0.5,
            'aggressive': 0.7
        }


class DataProcessor:
    """Enhanced data processing for multiple data sources"""
    
    def __init__(self, config: LendingConfig):
        self.config = config
        self.hmda_data = None
        self.census_data = None 
        self.hpi_data = None
        self.master_dataset = None
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load and validate all data sources"""
        
        logger.info("Loading all data sources...")
        
        # Load HMDA data (Kansas only)
        hmda_path = self.config.DATA_DIR / "state_KS.csv"
        if hmda_path.exists():
            self.hmda_data = pd.read_csv(hmda_path)
            logger.info(f"Loaded HMDA data: {len(self.hmda_data)} records")
        else:
            raise FileNotFoundError(f"HMDA data not found: {hmda_path}")
            
        # Load Census data
        census_path = self.config.DATA_DIR / "enhanced_census_data.csv"
        if census_path.exists():
            self.census_data = pd.read_csv(census_path)
            # Filter for Kansas (state code 20)
            self.census_data = self.census_data[
                self.census_data['state_code'] == int(self.config.KANSAS_FIPS_CODE)
            ]
            logger.info(f"Loaded Census data: {len(self.census_data)} Kansas tracts")
        else:
            raise FileNotFoundError(f"Census data not found: {census_path}")
            
        # Load HPI data  
        hpi_path = self.config.DATA_DIR / "hpi_at_tract.csv"
        if hpi_path.exists():
            self.hpi_data = pd.read_csv(hpi_path)
            # Filter for Kansas
            self.hpi_data = self.hpi_data[
                self.hpi_data['state_abbr'] == self.config.KANSAS_STATE_CODE
            ]
            logger.info(f"Loaded HPI data: {len(self.hpi_data)} Kansas records")
        else:
            logger.warning(f"HPI data not found: {hpi_path}")
            
        return {
            'hmda': self.hmda_data,
            'census': self.census_data,
            'hpi': self.hpi_data
        }
    
    def clean_hmda_data(self) -> pd.DataFrame:
        """Clean and prepare HMDA data"""
        
        logger.info("Cleaning HMDA data...")
        
        df = self.hmda_data.copy()
        
        # Handle missing values
        numeric_cols = ['loan_amount', 'income', 'debt_to_income_ratio', 
                       'applicant_credit_score_type', 'loan_to_value_ratio']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Use existing census_tract field and clean it
        # Filter out records with missing census tract
        df = df[df['census_tract'].notna()]
        df = df[df['census_tract'] != 'NA']
        
        # Convert census_tract to string and remove decimals (from float conversion)
        df['census_tract'] = df['census_tract'].astype(float).astype(int).astype(str)
        
        # Filter valid records
        df = df[df['action_taken'].isin(range(1, 9))]
        
        # Create binary target for loan approval
        df['loan_approved'] = df['action_taken'].isin([1, 2, 6, 8]).astype(int)
        df['loan_denied'] = (df['action_taken'] == 3).astype(int)
        
        logger.info(f"Cleaned HMDA data: {len(df)} valid records")
        return df
    
    def create_master_dataset(self) -> pd.DataFrame:
        """Create comprehensive dataset by merging all sources"""
        
        logger.info("Creating master dataset...")
        
        # Start with cleaned HMDA data
        hmda_clean = self.clean_hmda_data()
        
        # Aggregate HMDA data by census tract
        hmda_agg = hmda_clean.groupby('census_tract').agg({
            'loan_amount': ['count', 'mean', 'median'],
            'income': ['mean', 'median'],
            'debt_to_income_ratio': 'mean',
            'loan_to_value_ratio': 'mean',
            'loan_approved': ['sum', 'mean'],
            'loan_denied': ['sum', 'mean'],
            'action_taken': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 3
        }).round(2)
        
        # Flatten column names
        hmda_agg.columns = [
            'loan_count', 'avg_loan_amount', 'median_loan_amount',
            'avg_income', 'median_income', 'avg_dti', 'avg_ltv',
            'total_approved', 'approval_rate', 'total_denied', 'denial_rate',
            'most_common_action'
        ]
        
        hmda_agg = hmda_agg.reset_index()
        
        # Merge with census data
        census_clean = self.census_data.copy()
        census_clean['census_tract'] = census_clean['census_tract'].astype(str)
        
        master = pd.merge(
            hmda_agg, census_clean, 
            on='census_tract', 
            how='inner'
        )
        
        # Merge with HPI data (latest year)
        if self.hpi_data is not None:
            # Get latest HPI data for each tract
            hpi_latest = self.hpi_data.loc[
                self.hpi_data.groupby('census_tract')['year'].idxmax()
            ][['census_tract', 'hpi', 'annual_change']].copy()
            
            hpi_latest['census_tract'] = hpi_latest['census_tract'].astype(str)
            
            master = pd.merge(
                master, hpi_latest,
                on='census_tract',
                how='left'
            )
        
        self.master_dataset = master
        logger.info(f"Master dataset created: {len(master)} census tracts")
        
        return master


class OpportunityScoreCalculator:
    """Calculate meaningful opportunity scores based on market data"""
    
    def __init__(self, config: LendingConfig):
        self.config = config
        self.weights = config.OPPORTUNITY_WEIGHTS
        
    def calculate_market_accessibility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market accessibility score (0-100)"""
        
        # Normalize metrics to 0-100 scale
        def normalize_to_100(series, higher_better=True):
            if series.isna().all():
                return pd.Series(50, index=series.index)
            
            min_val, max_val = series.min(), series.max()
            if min_val == max_val:
                return pd.Series(50, index=series.index)
            
            normalized = (series - min_val) / (max_val - min_val) * 100
            return normalized if higher_better else 100 - normalized
        
        # Components of market accessibility
        population_score = normalize_to_100(df['total_population'], True)
        income_score = normalize_to_100(df['median_household_income'], True) 
        loan_activity_score = normalize_to_100(df['loan_count'], True)
        
        # Weighted average
        accessibility = (
            population_score * 0.4 +
            income_score * 0.35 + 
            loan_activity_score * 0.25
        )
        
        return accessibility.fillna(50)
    
    def calculate_risk_factors(self, df: pd.DataFrame) -> pd.Series:
        """Calculate risk factor score (0-100, higher = less risky)"""
        
        def normalize_to_100(series, higher_better=True):
            if series.isna().all():
                return pd.Series(50, index=series.index)
            
            min_val, max_val = series.min(), series.max()
            if min_val == max_val:
                return pd.Series(50, index=series.index)
            
            normalized = (series - min_val) / (max_val - min_val) * 100
            return normalized if higher_better else 100 - normalized
        
        # Risk components (lower = better for risk)
        unemployment_risk = normalize_to_100(df['unemployment_rate'], False)
        dti_risk = normalize_to_100(df['avg_dti'], False) if 'avg_dti' in df.columns else pd.Series(50, index=df.index)
        denial_risk = normalize_to_100(df['denial_rate'], False) if 'denial_rate' in df.columns else pd.Series(50, index=df.index)
        
        # Weighted risk score
        risk_score = (
            unemployment_risk * 0.4 +
            dti_risk * 0.35 +
            denial_risk * 0.25
        )
        
        return risk_score.fillna(50)
    
    def calculate_economic_indicators(self, df: pd.DataFrame) -> pd.Series:
        """Calculate economic strength indicators (0-100)"""
        
        def normalize_to_100(series, higher_better=True):
            if series.isna().all():
                return pd.Series(50, index=series.index)
            
            min_val, max_val = series.min(), series.max()
            if min_val == max_val:
                return pd.Series(50, index=series.index)
            
            normalized = (series - min_val) / (max_val - min_val) * 100
            return normalized if higher_better else 100 - normalized
        
        # Economic indicators
        income_strength = normalize_to_100(df['median_household_income'], True)
        labor_participation = normalize_to_100(df['civilian_labor_force'], True)
        
        # HPI growth if available
        hpi_growth = pd.Series(50, index=df.index)
        if 'annual_change' in df.columns:
            hpi_growth = normalize_to_100(df['annual_change'], True)
        
        economic_score = (
            income_strength * 0.4 +
            labor_participation * 0.35 +
            hpi_growth * 0.25
        )
        
        return economic_score.fillna(50)
    
    def calculate_lending_activity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate lending activity score (0-100)"""
        
        def normalize_to_100(series, higher_better=True):
            if series.isna().all():
                return pd.Series(50, index=series.index)
            
            min_val, max_val = series.min(), series.max()
            if min_val == max_val:
                return pd.Series(50, index=series.index)
            
            normalized = (series - min_val) / (max_val - min_val) * 100
            return normalized if higher_better else 100 - normalized
        
        # Lending activity components
        loan_volume = normalize_to_100(df['loan_count'], True) if 'loan_count' in df.columns else pd.Series(50, index=df.index)
        avg_loan_size = normalize_to_100(df['avg_loan_amount'], True) if 'avg_loan_amount' in df.columns else pd.Series(50, index=df.index)
        approval_rate = normalize_to_100(df['approval_rate'], True) if 'approval_rate' in df.columns else pd.Series(50, index=df.index)
        
        activity_score = (
            loan_volume * 0.4 +
            avg_loan_size * 0.35 +
            approval_rate * 0.25
        )
        
        return activity_score.fillna(50)
    
    def calculate_opportunity_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive opportunity score"""
        
        logger.info("Calculating opportunity scores...")
        
        result_df = df.copy()
        
        # Calculate component scores
        result_df['market_accessibility'] = self.calculate_market_accessibility(df)
        result_df['risk_factors'] = self.calculate_risk_factors(df)
        result_df['economic_indicators'] = self.calculate_economic_indicators(df)
        result_df['lending_activity'] = self.calculate_lending_activity(df)
        
        # Calculate weighted opportunity score
        result_df['opportunity_score'] = (
            result_df['market_accessibility'] * self.weights['market_accessibility'] +
            result_df['risk_factors'] * self.weights['risk_factors'] +
            result_df['economic_indicators'] * self.weights['economic_indicators'] +
            result_df['lending_activity'] * self.weights['lending_activity']
        ).round(2)
        
        # Classify opportunity levels
        def classify_opportunity(score):
            if score >= 75:
                return "High"
            elif score >= 50:
                return "Medium"
            else:
                return "Low"
        
        result_df['opportunity_level'] = result_df['opportunity_score'].apply(classify_opportunity)
        
        logger.info("Opportunity scores calculated successfully")
        return result_df


if __name__ == "__main__":
    # Initialize configuration
    config = LendingConfig()
    
    # Create necessary directories
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    
    # Initialize data processor
    processor = DataProcessor(config)
    
    # Load and process data
    try:
        data_sources = processor.load_all_data()
        master_data = processor.create_master_dataset()
        
        # Calculate opportunity scores
        score_calculator = OpportunityScoreCalculator(config)
        results = score_calculator.calculate_opportunity_score(master_data)
        
        logger.info("Analysis complete")
        
        # Display summary
        print("\n" + "="*60)
        print("KANSAS LENDING OPPORTUNITY ANALYSIS - SUMMARY")
        print("="*60)
        print(f"Total Census Tracts Analyzed: {len(results)}")
        print(f"Average Opportunity Score: {results['opportunity_score'].mean():.2f}")
        print("\nOpportunity Level Distribution:")
        print(results['opportunity_level'].value_counts())
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise