#!/usr/bin/env python3
"""
HMDA-specific Feature Engineering for Loan Outcome Prediction

This module provides comprehensive feature engineering specifically designed for HMDA data,
addressing the unique challenges of mortgage lending prediction including:
1. Advanced financial ratio calculations
2. Risk categorization and scoring
3. Demographic encoding for fair lending compliance
4. Synthetic credit score proxy creation
5. Denial reason analysis and prediction
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class HMDAFeatureEngineer:
    """Advanced feature engineering specifically for HMDA loan data"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
        
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create proper target variables for HMDA data
        
        HMDA Action Taken Codes:
        1 - Loan originated (APPROVED)
        2 - Application approved but not accepted (APPROVED)
        3 - Application denied (DENIED)
        4 - Application withdrawn by applicant (EXCLUDE)
        5 - File closed for incompleteness (EXCLUDE)
        6 - Purchased loan (APPROVED)
        7 - Preapproval request denied (DENIED)
        8 - Preapproval request approved but not accepted (APPROVED)
        """
        df_targets = df.copy()
        
        # Create comprehensive approval definition
        df_targets['loan_approved'] = df_targets['action_taken'].isin([1, 2, 6, 8]).astype(int)
        df_targets['loan_denied'] = df_targets['action_taken'].isin([3, 7]).astype(int)
        
        # Create binary target for modeling (1=approved, 0=denied)
        # Only include clear approve/deny cases for training
        df_targets['target'] = np.where(
            df_targets['action_taken'].isin([1, 2, 6, 8]), 1,
            np.where(df_targets['action_taken'].isin([3, 7]), 0, -1)
        )
        
        return df_targets
    
    def create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced financial ratio and risk features"""
        df_financial = df.copy()
        
        # 1. Income-based features (income is in thousands)
        if 'income' in df.columns and 'loan_amount' in df.columns:
            # Convert income to numeric, handling non-numeric values
            df_financial['income'] = pd.to_numeric(df_financial['income'], errors='coerce')
            df_financial['loan_amount'] = pd.to_numeric(df_financial['loan_amount'], errors='coerce')
            
            # Calculate actual income
            df_financial['actual_income'] = df_financial['income'] * 1000
            
            # Only calculate ratios where both values are valid
            valid_mask = (df_financial['income'] > 0) & (df_financial['loan_amount'] > 0)
            df_financial.loc[valid_mask, 'loan_to_income_ratio'] = (
                df_financial.loc[valid_mask, 'loan_amount'] / df_financial.loc[valid_mask, 'actual_income']
            )
            
            # Income adequacy categories
            df_financial['income_adequacy'] = pd.cut(
                df_financial['loan_to_income_ratio'],
                bins=[-np.inf, 3, 4, 5, np.inf],
                labels=['adequate', 'moderate_risk', 'high_risk', 'very_high_risk']
            )
            
            # Annual debt service to income (only if interest rate and loan term are available)
            if 'loan_term' in df.columns and 'interest_rate' in df.columns:
                # Convert to numeric
                df_financial['interest_rate'] = pd.to_numeric(df_financial['interest_rate'], errors='coerce')
                df_financial['loan_term'] = pd.to_numeric(df_financial['loan_term'], errors='coerce')
                
                # Only calculate for valid numeric values
                rate_mask = (df_financial['interest_rate'] > 0) & (df_financial['loan_term'] > 0)
                
                if rate_mask.any():
                    monthly_rate = (df_financial.loc[rate_mask, 'interest_rate'] / 100) / 12
                    n_payments = df_financial.loc[rate_mask, 'loan_term']
                    
                    # Calculate monthly payment using standard mortgage formula
                    numerator = monthly_rate * (1 + monthly_rate) ** n_payments
                    denominator = (1 + monthly_rate) ** n_payments - 1
                    
                    # Avoid division by zero
                    valid_payment_mask = denominator != 0
                    if valid_payment_mask.any():
                        df_financial.loc[rate_mask & valid_payment_mask, 'monthly_payment'] = (
                            df_financial.loc[rate_mask & valid_payment_mask, 'loan_amount'] * 
                            (numerator.loc[valid_payment_mask] / denominator.loc[valid_payment_mask])
                        )
                        
                        df_financial.loc[rate_mask, 'monthly_income'] = df_financial.loc[rate_mask, 'actual_income'] / 12
                        
                        # Avoid division by zero for payment to income ratio
                        valid_income_mask = df_financial.loc[rate_mask, 'monthly_income'] > 0
                        df_financial.loc[rate_mask & valid_income_mask, 'payment_to_income_ratio'] = (
                            df_financial.loc[rate_mask & valid_income_mask, 'monthly_payment'] / 
                            df_financial.loc[rate_mask & valid_income_mask, 'monthly_income']
                        )
        
        # 2. Property value and down payment features
        if 'property_value' in df.columns and 'loan_amount' in df.columns:
            # Convert to numeric
            df_financial['property_value'] = pd.to_numeric(df_financial['property_value'], errors='coerce')
            
            # Only calculate for valid values
            valid_mask = (df_financial['property_value'] > 0) & (df_financial['loan_amount'] > 0)
            
            if valid_mask.any():
                df_financial.loc[valid_mask, 'down_payment'] = (
                    df_financial.loc[valid_mask, 'property_value'] - df_financial.loc[valid_mask, 'loan_amount']
                )
                df_financial.loc[valid_mask, 'down_payment_ratio'] = (
                    df_financial.loc[valid_mask, 'down_payment'] / df_financial.loc[valid_mask, 'property_value']
                )
                df_financial.loc[valid_mask, 'loan_to_value_calculated'] = (
                    df_financial.loc[valid_mask, 'loan_amount'] / df_financial.loc[valid_mask, 'property_value']
                )
                
                # Down payment adequacy
                df_financial['down_payment_adequacy'] = pd.cut(
                    df_financial['down_payment_ratio'],
                    bins=[-np.inf, 0.05, 0.10, 0.20, np.inf],
                    labels=['minimal', 'low', 'conventional', 'substantial']
                )
        
        # 3. Enhanced DTI risk categories
        if 'debt_to_income_ratio' in df.columns:
            # Convert to numeric, handling percentage strings and other formats
            dti_series = df_financial['debt_to_income_ratio'].astype(str)
            
            # Handle percentage ranges like "20%-<30%", ">60%", "<20%", "50%-60%", etc.
            def parse_dti_range(dti_str):
                if pd.isna(dti_str) or dti_str in ['nan', 'NA', '', 'Exempt']:
                    return np.nan
                
                try:
                    # Convert to string and remove any whitespace
                    dti_str = str(dti_str).strip()
                    
                    # Handle special cases first
                    if dti_str.lower() == 'exempt':
                        return np.nan
                    
                    # Remove percentage signs
                    dti_str = dti_str.replace('%', '')
                    
                    # Handle ">X" format (e.g., ">60")
                    if dti_str.startswith('>'):
                        value = float(dti_str[1:])
                        # For >60%, use 65% as representative value
                        return value + 5
                    
                    # Handle "<X" format (e.g., "<20")  
                    if dti_str.startswith('<'):
                        value = float(dti_str[1:])
                        # For <20%, use 15% as representative value
                        return max(0, value - 5)
                    
                    # Handle "X%-<Y%" format (e.g., "20-<30")
                    if '-<' in dti_str:
                        parts = dti_str.split('-<')
                        if len(parts) == 2:
                            start = float(parts[0])
                            end = float(parts[1])
                            return (start + end) / 2
                    
                    # Handle "X%-Y%" format (e.g., "50-60")
                    if '-' in dti_str and not dti_str.startswith('-'):
                        parts = dti_str.split('-')
                        if len(parts) == 2:
                            try:
                                start = float(parts[0])
                                end = float(parts[1])
                                return (start + end) / 2
                            except ValueError:
                                pass
                    
                    # Handle direct numeric values
                    return float(dti_str)
                    
                except (ValueError, TypeError):
                    return np.nan
            
            df_financial['debt_to_income_ratio'] = dti_series.apply(parse_dti_range)
            
            df_financial['dti_risk_category'] = pd.cut(
                df_financial['debt_to_income_ratio'],
                bins=[-np.inf, 28, 36, 43, 50, np.inf],
                labels=['excellent', 'good', 'acceptable', 'risky', 'very_risky']
            )
            
            # DTI compliance flags
            df_financial['dti_compliant'] = (df_financial['debt_to_income_ratio'] <= 43).astype(int)
            df_financial['dti_qm_compliant'] = (df_financial['debt_to_income_ratio'] <= 43).astype(int)  # QM rule
        
        # 4. Loan size categories
        if 'loan_amount' in df.columns:
            df_financial['loan_size_category'] = pd.cut(
                df_financial['loan_amount'],
                bins=[-np.inf, 150000, 300000, 500000, 766550, np.inf],  # Using 2024 conforming loan limit
                labels=['small', 'medium', 'large', 'jumbo', 'super_jumbo']
            )
        
        # 5. Combined risk score
        risk_components = []
        
        if 'debt_to_income_ratio' in df_financial.columns:
            # Higher DTI = higher risk (normalize to 0-1)
            dti_risk = np.clip(df_financial['debt_to_income_ratio'] / 60, 0, 1)
            risk_components.append(dti_risk)
        
        if 'loan_to_value_calculated' in df_financial.columns:
            # Higher LTV = higher risk
            ltv_risk = np.clip(df_financial['loan_to_value_calculated'], 0, 1)
            risk_components.append(ltv_risk)
        
        if 'loan_to_income_ratio' in df_financial.columns:
            # Higher LTI = higher risk (normalize to 0-1, cap at 6)
            lti_risk = np.clip(df_financial['loan_to_income_ratio'] / 6, 0, 1)
            risk_components.append(lti_risk)
        
        if risk_components:
            df_financial['financial_risk_score'] = np.nanmean(risk_components, axis=0)
            df_financial['financial_risk_category'] = pd.cut(
                df_financial['financial_risk_score'],
                bins=[-np.inf, 0.3, 0.5, 0.7, np.inf],
                labels=['low_risk', 'moderate_risk', 'high_risk', 'very_high_risk']
            )
        
        # Clean infinite and extreme values
        numeric_cols = df_financial.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Replace infinite values with NaN
            df_financial[col] = df_financial[col].replace([np.inf, -np.inf], np.nan)
            
            # Cap extremely large values (beyond 99.9th percentile)
            if df_financial[col].notna().sum() > 0:
                q999 = df_financial[col].quantile(0.999)
                q001 = df_financial[col].quantile(0.001)
                if pd.notna(q999) and pd.notna(q001):
                    df_financial[col] = df_financial[col].clip(lower=q001, upper=q999)
        
        return df_financial
    
    def create_synthetic_credit_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create synthetic credit score proxy since HMDA doesn't include credit scores
        Based on observable financial behavior indicators
        """
        df_credit = df.copy()
        
        credit_indicators = []
        
        # 1. DTI indicator (lower is better)
        if 'debt_to_income_ratio' in df.columns:
            dti_normalized = 1 - np.clip(df_credit['debt_to_income_ratio'] / 60, 0, 1)
            credit_indicators.append(dti_normalized * 0.3)  # 30% weight
        
        # 2. Down payment indicator (higher is better)
        if 'down_payment_ratio' in df_credit.columns:
            dp_normalized = np.clip(df_credit['down_payment_ratio'] * 2, 0, 1)
            credit_indicators.append(dp_normalized * 0.25)  # 25% weight
        
        # 3. Income stability indicator
        if 'loan_to_income_ratio' in df_credit.columns:
            lti_normalized = 1 - np.clip(df_credit['loan_to_income_ratio'] / 5, 0, 1)
            credit_indicators.append(lti_normalized * 0.25)  # 25% weight
        
        # 4. Property type and loan purpose risk
        if 'loan_purpose' in df.columns:
            # 1=purchase (medium risk), 2=improvement (high risk), 3=refinance (low risk)
            purpose_score = df_credit['loan_purpose'].map({1: 0.6, 2: 0.3, 3: 0.8})
            credit_indicators.append(purpose_score.fillna(0.5) * 0.1)  # 10% weight
        
        # 5. Occupancy type indicator
        if 'occupancy_type' in df.columns:
            # 1=owner occupied (low risk), 2=not owner occupied (high risk)
            occupancy_score = df_credit['occupancy_type'].map({1: 0.8, 2: 0.4})
            credit_indicators.append(occupancy_score.fillna(0.6) * 0.1)  # 10% weight
        
        if credit_indicators:
            # Combine indicators
            credit_score_normalized = np.sum(credit_indicators, axis=0)
            
            # Convert to FICO-like scale (300-850)
            df_credit['credit_score_proxy'] = 300 + (credit_score_normalized * 550)
            
            # Create credit score categories
            df_credit['credit_score_category'] = pd.cut(
                df_credit['credit_score_proxy'],
                bins=[0, 580, 620, 660, 720, 850],
                labels=['poor', 'fair', 'good', 'very_good', 'excellent']
            )
        
        return df_credit
    
    def create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create demographic features while maintaining fair lending compliance"""
        df_demo = df.copy()
        
        # Encode categorical demographics
        demographic_cols = [
            'applicant_race-1', 'applicant_ethnicity', 'applicant_sex',
            'co-applicant_race-1', 'co-applicant_ethnicity', 'co-applicant_sex'
        ]
        
        for col in demographic_cols:
            if col in df.columns:
                # Create encoded version
                encoded_col = f'{col}_encoded'
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_demo[encoded_col] = self.encoders[col].fit_transform(df_demo[col].astype(str))
                else:
                    # Handle unseen labels gracefully
                    try:
                        df_demo[encoded_col] = self.encoders[col].transform(df_demo[col].astype(str))
                    except ValueError:
                        # For unseen labels, assign a default value (0)
                        known_values = set(self.encoders[col].classes_)
                        df_demo[encoded_col] = df_demo[col].astype(str).apply(
                            lambda x: self.encoders[col].transform([x])[0] if x in known_values else 0
                        )
        
        # Age category features
        if 'applicant_age' in df.columns:
            df_demo['applicant_age_category'] = pd.cut(
                pd.to_numeric(df_demo['applicant_age'], errors='coerce'),
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=['young', 'early_career', 'mid_career', 'late_career', 'pre_retirement', 'retirement']
            )
        
        # Joint application indicator
        if 'co-applicant_sex' in df.columns:
            df_demo['joint_application'] = (df_demo['co-applicant_sex'] != '5').astype(int)  # 5 = not applicable
        
        return df_demo
    
    def create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create geographic and market-based features"""
        df_geo = df.copy()
        
        # Encode geographic identifiers
        geo_cols = ['state_code', 'county_code', 'census_tract', 'derived_msa-md']
        
        for col in geo_cols:
            if col in df.columns:
                encoded_col = f'{col}_encoded'
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_geo[encoded_col] = self.encoders[col].fit_transform(df_geo[col].astype(str))
                else:
                    # Handle unseen labels gracefully
                    try:
                        df_geo[encoded_col] = self.encoders[col].transform(df_geo[col].astype(str))
                    except ValueError:
                        # For unseen labels, assign a default value (0)
                        known_values = set(self.encoders[col].classes_)
                        df_geo[encoded_col] = df_geo[col].astype(str).apply(
                            lambda x: self.encoders[col].transform([x])[0] if x in known_values else 0
                        )
        
        # Census tract income features
        if 'tract_to_msa_income_percentage' in df.columns:
            df_geo['tract_income_category'] = pd.cut(
                df_geo['tract_to_msa_income_percentage'],
                bins=[0, 80, 100, 120, np.inf],
                labels=['low_income_area', 'moderate_income_area', 'middle_income_area', 'high_income_area']
            )
        
        # Minority population concentration
        if 'tract_minority_population_percent' in df.columns:
            df_geo['minority_concentration'] = pd.cut(
                df_geo['tract_minority_population_percent'],
                bins=[0, 10, 30, 50, 100],
                labels=['low_minority', 'moderate_minority', 'high_minority', 'majority_minority']
            )
        
        return df_geo
    
    def create_denial_reason_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to denial reasons for analysis"""
        df_denial = df.copy()
        
        denial_cols = [col for col in df.columns if 'denial_reason' in col.lower()]
        
        if denial_cols:
            # Convert denial reason columns to numeric, treating non-numeric as 0
            for col in denial_cols:
                df_denial[col] = pd.to_numeric(df_denial[col], errors='coerce').fillna(0)
            
            # Count number of denial reasons
            df_denial['num_denial_reasons'] = (df_denial[denial_cols] > 0).sum(axis=1)
            df_denial['has_denial_reason'] = (df_denial['num_denial_reasons'] > 0).astype(int)
            
            # Create binary flags for each denial reason type
            denial_reason_map = {
                1: 'denial_debt_to_income',
                2: 'denial_employment_history',
                3: 'denial_credit_history',
                4: 'denial_collateral',
                5: 'denial_insufficient_cash',
                6: 'denial_unverifiable_info',
                7: 'denial_incomplete_application',
                8: 'denial_mortgage_insurance',
                9: 'denial_other',
                10: 'denial_not_applicable'
            }
            
            for reason_code, reason_name in denial_reason_map.items():
                df_denial[reason_name] = ((df_denial[denial_cols] == reason_code).any(axis=1)).astype(int)
        
        return df_denial
    
    def create_loan_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to loan products and terms"""
        df_product = df.copy()
        
        # Loan type risk assessment
        if 'loan_type' in df.columns:
            # Convert to numeric first
            df_product['loan_type'] = pd.to_numeric(df_product['loan_type'], errors='coerce')
            # 1=conventional, 2=FHA, 3=VA, 4=USDA
            loan_type_risk = df_product['loan_type'].map({
                1: 'medium_risk',  # Conventional
                2: 'low_risk',     # FHA (government backed)
                3: 'low_risk',     # VA (government backed)
                4: 'low_risk'      # USDA (government backed)
            })
            df_product['loan_type_risk'] = loan_type_risk.fillna('unknown')
        
        # Interest rate features
        if 'interest_rate' in df.columns:
            # Convert to numeric
            df_product['interest_rate'] = pd.to_numeric(df_product['interest_rate'], errors='coerce')
            
            df_product['interest_rate_category'] = pd.cut(
                df_product['interest_rate'],
                bins=[0, 3, 5, 7, 10, np.inf],
                labels=['very_low', 'low', 'moderate', 'high', 'very_high']
            )
        
        # Rate spread features
        if 'rate_spread' in df.columns:
            df_product['rate_spread'] = pd.to_numeric(df_product['rate_spread'], errors='coerce')
            
            df_product['rate_spread_category'] = pd.cut(
                df_product['rate_spread'],
                bins=[-np.inf, 0, 1.5, 3, np.inf],
                labels=['below_market', 'at_market', 'above_market', 'high_cost']
            )
        
        # Loan term features
        if 'loan_term' in df.columns:
            df_product['loan_term'] = pd.to_numeric(df_product['loan_term'], errors='coerce')
            
            df_product['loan_term_category'] = pd.cut(
                df_product['loan_term'],
                bins=[0, 180, 360, 480, np.inf],
                labels=['short_term', 'standard', 'long_term', 'very_long_term']
            )
        
        # Handle loan_to_value_ratio properly
        if 'loan_to_value_ratio' in df.columns:
            # Convert to numeric, handling various formats
            ltv_series = df_product['loan_to_value_ratio'].astype(str)
            
            def parse_ltv(ltv_str):
                if pd.isna(ltv_str) or ltv_str in ['nan', 'NA', '', 'Exempt']:
                    return np.nan
                
                try:
                    # Remove percentage signs and convert
                    ltv_str = str(ltv_str).replace('%', '')
                    ltv_value = float(ltv_str)
                    
                    # If value is greater than 1, assume it's already a percentage
                    if ltv_value > 1:
                        return ltv_value / 100
                    else:
                        return ltv_value
                        
                except (ValueError, TypeError):
                    return np.nan
            
            df_product['loan_to_value_ratio'] = ltv_series.apply(parse_ltv)
        
        return df_product
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations"""
        
        # Start with target variables
        df_engineered = self.create_target_variables(df)
        
        # Apply all feature engineering
        df_engineered = self.create_financial_features(df_engineered)
        df_engineered = self.create_synthetic_credit_score(df_engineered)
        df_engineered = self.create_demographic_features(df_engineered)
        df_engineered = self.create_geographic_features(df_engineered)
        df_engineered = self.create_denial_reason_features(df_engineered)
        df_engineered = self.create_loan_product_features(df_engineered)
        
        return df_engineered
    
    def prepare_for_modeling(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare engineered features for machine learning - using focused 17-feature set"""
        
        # Define the core 17 features that match prediction pipeline
        core_features = [
            'debt_to_income_ratio', 'loan_to_value_calculated', 'loan_to_income_ratio',
            'down_payment_ratio', 'credit_score_proxy', 'financial_risk_score',
            'monthly_payment', 'payment_to_income_ratio', 'age_category',
            'applicant_race_encoded', 'applicant_sex_encoded',
            'dti_risk_category_encoded', 'ltv_risk_category_encoded', 
            'loan_size_category_encoded', 'income_category_encoded', 
            'credit_score_category_encoded', 'financial_risk_category_encoded'
        ]
        
        # Use only the core features that exist in the dataframe
        feature_columns = [col for col in core_features if col in df.columns]
        
        # Handle missing values and infinite values
        df_clean = df.copy()
        
        for col in feature_columns:
            if df_clean[col].isnull().sum() > 0:
                if col in ['income', 'loan_amount', 'property_value']:
                    # Use median for key financial features
                    median_val = df_clean[col].median()
                    if pd.isna(median_val) or not np.isfinite(median_val):
                        # Set reasonable defaults
                        defaults = {
                            'income': 75,
                            'loan_amount': 200000,
                            'property_value': 250000
                        }
                        median_val = defaults.get(col, 0)
                    df_clean[col].fillna(median_val, inplace=True)
                else:
                    # Use mode for categorical features or 0 for others
                    mode_val = df_clean[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 and np.isfinite(mode_val[0]) else 0
                    df_clean[col].fillna(fill_val, inplace=True)
            
            # Handle infinite values
            if np.isinf(df_clean[col]).any():
                # Replace infinite values with finite values
                finite_values = df_clean[col][np.isfinite(df_clean[col])]
                if len(finite_values) > 0:
                    # Replace +inf with 99th percentile, -inf with 1st percentile
                    p99 = finite_values.quantile(0.99)
                    p01 = finite_values.quantile(0.01)
                    
                    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], [p99, p01])
                else:
                    # If all values are infinite, replace with 0
                    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], 0)
            
            # Ensure no remaining NaN or infinite values
            if df_clean[col].isnull().any() or np.isinf(df_clean[col]).any():
                df_clean[col] = df_clean[col].fillna(0)
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], 0)
        
        # Handle categorical features encoded as strings
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        exclude_categorical = ['action_taken', 'target', 'loan_approved', 'loan_denied']
        for col in categorical_features:
            if col not in exclude_categorical:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_clean[f'{col}_encoded'] = self.encoders[col].fit_transform(df_clean[col].astype(str))
                else:
                    # Handle unseen labels gracefully
                    try:
                        df_clean[f'{col}_encoded'] = self.encoders[col].transform(df_clean[col].astype(str))
                    except ValueError:
                        # For unseen labels, assign a default value (0)
                        known_values = set(self.encoders[col].classes_)
                        df_clean[f'{col}_encoded'] = df_clean[col].astype(str).apply(
                            lambda x: self.encoders[col].transform([x])[0] if x in known_values else 0
                        )
                
                feature_columns.append(f'{col}_encoded')
        
        # Final validation - ensure all feature columns are numeric and finite
        for col in feature_columns:
            if col in df_clean.columns:
                # Convert to numeric if not already
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Fill any remaining NaN values
                df_clean[col] = df_clean[col].fillna(0)
                
                # Replace any remaining infinite values
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], 0)
                
                # Ensure the column is finite
                if not np.isfinite(df_clean[col]).all():
                    logger.warning(f"Column {col} still contains non-finite values, replacing with 0")
                    df_clean[col] = np.where(np.isfinite(df_clean[col]), df_clean[col], 0)
        
        return df_clean, feature_columns
    
    def engineer_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a focused set of features for prediction using only the available input columns.
        This method focuses on the core financial features that are typically available
        in a loan application: debt_to_income_ratio, loan_to_value_ratio, property_value, 
        loan_amount, income
        """
        df_features = df.copy()
        
        # Ensure required columns exist with reasonable defaults
        required_columns = {
            'debt_to_income_ratio': 30.0,
            'loan_to_value_ratio': 80.0,
            'property_value': 300000,
            'loan_amount': 250000,
            'income': 75,  # in thousands
            'interest_rate': 6.5,
            'loan_term': 360,
            'applicant_age': 35,
            'applicant_race-1': 5,
            'applicant_sex': 1,
            'county_code': 20091,
            'census_tract': 20091001400
        }
        
        # Add missing columns with defaults
        for col, default_val in required_columns.items():
            if col not in df_features.columns:
                df_features[col] = default_val
        
        # Convert to numeric, handling any string values
        numeric_cols = ['debt_to_income_ratio', 'loan_to_value_ratio', 'property_value', 
                       'loan_amount', 'income', 'interest_rate', 'loan_term', 'applicant_age']
        
        for col in numeric_cols:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
        
        # 1. Core Financial Ratios
        df_features['actual_income'] = df_features['income'] * 1000  # Convert to dollars
        
        # Loan-to-income ratio
        valid_mask = (df_features['income'] > 0) & (df_features['loan_amount'] > 0)
        df_features.loc[valid_mask, 'loan_to_income_ratio'] = (
            df_features.loc[valid_mask, 'loan_amount'] / df_features.loc[valid_mask, 'actual_income']
        )
        df_features['loan_to_income_ratio'] = df_features['loan_to_income_ratio'].fillna(0)
        
        # Calculate LTV if not provided or verify provided value
        if 'property_value' in df_features.columns and df_features['property_value'].notna().any():
            valid_pv_mask = (df_features['property_value'] > 0) & (df_features['loan_amount'] > 0)
            calculated_ltv = (df_features.loc[valid_pv_mask, 'loan_amount'] / 
                            df_features.loc[valid_pv_mask, 'property_value'] * 100)
            df_features.loc[valid_pv_mask, 'loan_to_value_calculated'] = calculated_ltv
        else:
            df_features['loan_to_value_calculated'] = df_features['loan_to_value_ratio']
        
        df_features['loan_to_value_calculated'] = df_features['loan_to_value_calculated'].fillna(80)
        
        # Down payment calculations
        df_features['down_payment_amount'] = (df_features['property_value'] * 
                                            (1 - df_features['loan_to_value_calculated'] / 100))
        df_features['down_payment_ratio'] = 100 - df_features['loan_to_value_calculated']
        
        # 2. Risk Scoring
        # DTI Risk Categories
        df_features['dti_risk_category'] = pd.cut(
            df_features['debt_to_income_ratio'],
            bins=[0, 20, 28, 36, 45, 100],
            labels=['Excellent', 'Good', 'Acceptable', 'High', 'Very High'],
            include_lowest=True
        ).astype(str)
        
        # LTV Risk Categories
        df_features['ltv_risk_category'] = pd.cut(
            df_features['loan_to_value_calculated'],
            bins=[0, 80, 90, 95, 100],
            labels=['Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        ).astype(str)
        
        # Loan size categories
        df_features['loan_size_category'] = pd.cut(
            df_features['loan_amount'],
            bins=[0, 200000, 400000, 600000, 1000000, float('inf')],
            labels=['Small', 'Medium', 'Large', 'Jumbo', 'Super Jumbo'],
            include_lowest=True
        ).astype(str)
        
        # Income categories
        df_features['income_category'] = pd.cut(
            df_features['actual_income'],
            bins=[0, 50000, 75000, 100000, 150000, float('inf')],
            labels=['Low', 'Lower Middle', 'Middle', 'Upper Middle', 'High'],
            include_lowest=True
        ).astype(str)
        
        # 3. Synthetic Credit Score Proxy
        credit_score_proxy = 850  # Start with perfect score
        
        # DTI impact
        credit_score_proxy -= np.where(df_features['debt_to_income_ratio'] > 43, 100,
                                     np.where(df_features['debt_to_income_ratio'] > 36, 50,
                                            np.where(df_features['debt_to_income_ratio'] > 28, 25, 0)))
        
        # LTV impact
        credit_score_proxy -= np.where(df_features['loan_to_value_calculated'] > 95, 75,
                                     np.where(df_features['loan_to_value_calculated'] > 90, 50,
                                            np.where(df_features['loan_to_value_calculated'] > 80, 25, 0)))
        
        # Loan-to-income impact
        credit_score_proxy -= np.where(df_features['loan_to_income_ratio'] > 5, 100,
                                     np.where(df_features['loan_to_income_ratio'] > 4, 50,
                                            np.where(df_features['loan_to_income_ratio'] > 3, 25, 0)))
        
        df_features['credit_score_proxy'] = np.clip(credit_score_proxy, 300, 850)
        
        # Credit score categories
        df_features['credit_score_category'] = pd.cut(
            df_features['credit_score_proxy'],
            bins=[300, 580, 670, 740, 800, 850],
            labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'],
            include_lowest=True
        ).astype(str)
        
        # 4. Financial Risk Score (0-100)
        risk_score = 0
        
        # DTI component (40% weight)
        dti_score = np.clip(df_features['debt_to_income_ratio'] * 2, 0, 100)
        risk_score += dti_score * 0.4
        
        # LTV component (35% weight)
        ltv_score = np.clip(df_features['loan_to_value_calculated'] * 1.2, 0, 100)
        risk_score += ltv_score * 0.35
        
        # Loan-to-income component (25% weight)
        lti_score = np.clip(df_features['loan_to_income_ratio'] * 20, 0, 100)
        risk_score += lti_score * 0.25
        
        df_features['financial_risk_score'] = np.clip(risk_score, 0, 100)
        
        # Risk categories
        df_features['financial_risk_category'] = pd.cut(
            df_features['financial_risk_score'],
            bins=[0, 25, 50, 75, 100],
            labels=['Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        ).astype(str)
        
        # 5. Monthly Payment Calculations
        if 'interest_rate' in df_features.columns and 'loan_term' in df_features.columns:
            monthly_rate = df_features['interest_rate'] / 100 / 12
            n_payments = df_features['loan_term']
            
            # Monthly payment calculation
            payment_factor = (monthly_rate * (1 + monthly_rate) ** n_payments) / \
                           ((1 + monthly_rate) ** n_payments - 1)
            df_features['monthly_payment'] = df_features['loan_amount'] * payment_factor
            
            # Payment-to-income ratio
            monthly_income = df_features['actual_income'] / 12
            df_features['payment_to_income_ratio'] = (df_features['monthly_payment'] / monthly_income) * 100
        else:
            df_features['monthly_payment'] = 0
            df_features['payment_to_income_ratio'] = 0
        
        # 6. Basic Demographic Features (encoded as numeric)
        # Age categories
        if 'applicant_age' in df_features.columns:
            df_features['age_category'] = pd.cut(
                df_features['applicant_age'],
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=[1, 2, 3, 4, 5, 6],  # Numeric labels for easy encoding
                include_lowest=True
            ).astype(float)
        else:
            df_features['age_category'] = 3  # Default to middle-age category
        
        # Race and sex encoding (keep as numeric)
        df_features['applicant_race_encoded'] = pd.to_numeric(df_features.get('applicant_race-1', 5), errors='coerce')
        df_features['applicant_sex_encoded'] = pd.to_numeric(df_features.get('applicant_sex', 1), errors='coerce')
        
        # 7. Select final feature set
        feature_columns = [
            'debt_to_income_ratio', 'loan_to_value_calculated', 'loan_to_income_ratio',
            'down_payment_ratio', 'credit_score_proxy', 'financial_risk_score',
            'monthly_payment', 'payment_to_income_ratio', 'age_category',
            'applicant_race_encoded', 'applicant_sex_encoded'
        ]
        
        # Add categorical features as encoded versions
        categorical_features = ['dti_risk_category', 'ltv_risk_category', 'loan_size_category',
                              'income_category', 'credit_score_category', 'financial_risk_category']
        
        for cat_col in categorical_features:
            if cat_col not in self.encoders:
                self.encoders[cat_col] = LabelEncoder()
                # Fit on common categories
                common_categories = {
                    'dti_risk_category': ['Excellent', 'Good', 'Acceptable', 'High', 'Very High'],
                    'ltv_risk_category': ['Low', 'Medium', 'High', 'Very High'],
                    'loan_size_category': ['Small', 'Medium', 'Large', 'Jumbo', 'Super Jumbo'],
                    'income_category': ['Low', 'Lower Middle', 'Middle', 'Upper Middle', 'High'],
                    'credit_score_category': ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'],
                    'financial_risk_category': ['Low', 'Medium', 'High', 'Very High']
                }
                self.encoders[cat_col].fit(common_categories[cat_col])
            
            # Transform with fallback for unknown categories
            try:
                df_features[f'{cat_col}_encoded'] = self.encoders[cat_col].transform(df_features[cat_col])
            except ValueError:
                # Handle unknown categories by assigning median value
                median_val = len(self.encoders[cat_col].classes_) // 2
                df_features[f'{cat_col}_encoded'] = median_val
            
            feature_columns.append(f'{cat_col}_encoded')
        
        # Fill any remaining NaN values
        for col in feature_columns:
            if col in df_features.columns:
                df_features[col] = df_features[col].fillna(0)
                # Ensure finite values
                df_features[col] = np.where(np.isfinite(df_features[col]), df_features[col], 0)
        
        return df_features[feature_columns]

    def get_feature_importance_ranking(self) -> Dict[str, float]:
        """Return feature importance ranking for interpretability"""
        
        return {
            # Financial Features (High Priority)
            'debt_to_income_ratio': 0.95,
            'financial_risk_score': 0.90,
            'loan_to_value_calculated': 0.85,
            'loan_to_income_ratio': 0.80,
            'credit_score_proxy': 0.75,
            'down_payment_ratio': 0.70,
            'payment_to_income_ratio': 0.65,
            
            # Risk Categories (Medium-High Priority)
            'dti_risk_category': 0.60,
            'financial_risk_category': 0.55,
            'credit_score_category': 0.50,
            'loan_size_category': 0.45,
            
            # Product Features (Medium Priority)
            'loan_type_risk': 0.40,
            'interest_rate': 0.35,
            'loan_term': 0.30,
            'rate_spread': 0.25,
            
            # Geographic Features (Low-Medium Priority)
            'tract_income_category': 0.20,
            'minority_concentration': 0.15,
            
            # Demographic Features (Low Priority - for compliance monitoring)
            'applicant_age_category': 0.10,
            'joint_application': 0.05
        }