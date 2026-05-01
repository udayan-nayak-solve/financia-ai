"""
Data Processing and Feature Engineering Module
Handles data loading, cleaning, and feature creation for loan prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, List, Optional
import logging
from pathlib import Path
import joblib

from config_manager import get_config


class DataProcessor:
    """Handles data loading, cleaning, and preprocessing"""
    
    def __init__(self):
        """Initialize data processor with configuration"""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.scaler = None
        self.feature_columns = None
        self.categorical_encoders = {}
        self.categorical_categories = {}
        
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load loan data from CSV file
        
        Args:
            file_path: Optional custom file path
            
        Returns:
            Loaded DataFrame
        """
        if file_path is None:
            file_path = self.config.get('data.input_file')
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        self.logger.info(f"Loading data from: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean input data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Validating and cleaning data...")
        
        initial_rows = len(df)
        
        # Check for required columns
        required_columns = self.config.get('features.core_features', [])
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove rows with missing values in critical columns
        critical_columns = ['action_taken', 'income', 'loan_amount', 'credit_score']
        df = df.dropna(subset=critical_columns)
        
        # Remove outliers using IQR method
        df = self._remove_outliers(df)
        
        # Validate data ranges
        df = self._validate_ranges(df)
        
        final_rows = len(df)
        removed_rows = initial_rows - final_rows
        
        if removed_rows > 0:
            self.logger.warning(f"Removed {removed_rows} rows during validation ({removed_rows/initial_rows*100:.2f}%)")
        
        self.logger.info(f"Data validation completed. Final shape: {df.shape}")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        numeric_columns = ['income', 'loan_amount', 'property_value', 'credit_score']
        
        for col in numeric_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                before_count = len(df)
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                after_count = len(df)
                
                if before_count != after_count:
                    self.logger.debug(f"Removed {before_count - after_count} outliers from {col}")
        
        return df
    
    def _validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data ranges based on business rules"""
        # Income validation (in thousands)
        df = df[(df['income'] >= 10) & (df['income'] <= 1000)]
        
        # Credit score validation
        df = df[(df['credit_score'] >= 300) & (df['credit_score'] <= 850)]
        
        # DTI validation
        if 'debt_to_income_ratio' in df.columns:
            df = df[(df['debt_to_income_ratio'] >= 0) & (df['debt_to_income_ratio'] <= 100)]
        
        # LTV validation
        if 'loan_to_value_ratio' in df.columns:
            df = df[(df['loan_to_value_ratio'] >= 10) & (df['loan_to_value_ratio'] <= 100)]
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features for model training
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        self.logger.info("Creating derived features...")
        
        df = df.copy()
        
        # Loan to income ratio
        df['loan_to_income_ratio'] = df['loan_amount'] / (df['income'] * 1000)
        
        # Property to income ratio
        df['property_to_income_ratio'] = df['property_value'] / (df['income'] * 1000)
        
        # Credit score categories
        df['credit_score_category'] = pd.cut(
            df['credit_score'],
            bins=[0, 580, 670, 740, 800, 850],
            labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
        )
        
        # Income categories
        df['income_category'] = pd.cut(
            df['income'],
            bins=[0, 50, 100, 200, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Loan amount categories
        df['loan_amount_category'] = pd.cut(
            df['loan_amount'],
            bins=[0, 200000, 400000, 750000, float('inf')],
            labels=['Small', 'Medium', 'Large', 'Jumbo']
        )
        
        # Risk score (composite risk indicator)
        df['risk_score'] = self._calculate_risk_score(df)
        
        # Age of loan (years since activity)
        current_year = 2025
        df['loan_age_years'] = current_year - df['activity_year']
        
        # Binary indicators
        df['high_ltv'] = (df['loan_to_value_ratio'] > 90).astype(int)
        df['high_dti'] = (df['debt_to_income_ratio'] > 43).astype(int)
        df['low_credit'] = (df['credit_score'] < 620).astype(int)
        
        self.logger.info(f"Created derived features. New shape: {df.shape}")
        
        return df
    
    def _calculate_risk_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite risk score"""
        # Normalize components to 0-100 scale
        credit_component = (850 - df['credit_score']) / 550 * 40  # 40% weight
        dti_component = np.minimum(df['debt_to_income_ratio'] / 50 * 35, 35)  # 35% weight
        ltv_component = np.maximum((df['loan_to_value_ratio'] - 50) / 50 * 25, 0)  # 25% weight
        
        risk_score = credit_component + dti_component + ltv_component
        return np.clip(risk_score, 0, 100)
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'action_taken', is_training: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare features for model training
        
        Args:
            df: Input DataFrame
            target_column: Target variable column name (optional for prediction)
            is_training: Whether this is for training (fit encoders) or prediction (transform only)
            
        Returns:
            Tuple of (features DataFrame, target Series or None)
        """
        self.logger.info("Preparing features for modeling...")
        
        # Get feature columns
        core_features = self.config.get('features.core_features', [])
        derived_features = self.config.get('features.derived_features', [])
        
        # Select categorical features for encoding
        categorical_features = ['credit_score_category', 'income_category', 'loan_amount_category']
        
        # Create feature matrix
        feature_df = df.copy()
        
        # Handle categorical features with consistent encoding
        if is_training:
            # Fit encoders and store categories during training
            for cat_col in categorical_features:
                if cat_col in feature_df.columns:
                    # Store the categories
                    self.categorical_categories[cat_col] = feature_df[cat_col].cat.categories.tolist()
                    
                    # Create dummy variables with all categories
                    encoded = pd.get_dummies(
                        feature_df[cat_col], 
                        prefix=cat_col, 
                        drop_first=True,
                        dtype=int
                    )
                    
                    # Store the column names for consistent ordering
                    self.categorical_encoders[cat_col] = encoded.columns.tolist()
                    
                    feature_df = pd.concat([feature_df, encoded], axis=1)
                    feature_df.drop(cat_col, axis=1, inplace=True)
        else:
            # Transform using stored encoders for prediction
            for cat_col in categorical_features:
                if cat_col in feature_df.columns and cat_col in self.categorical_categories:
                    # Ensure all categories are represented, even if not in current data
                    cat_series = feature_df[cat_col].astype('category')
                    cat_series = cat_series.cat.set_categories(self.categorical_categories[cat_col])
                    
                    # Create dummy variables
                    encoded = pd.get_dummies(
                        cat_series, 
                        prefix=cat_col, 
                        drop_first=True,
                        dtype=int
                    )
                    
                    # Ensure consistent column ordering
                    if cat_col in self.categorical_encoders:
                        expected_cols = self.categorical_encoders[cat_col]
                        for col in expected_cols:
                            if col not in encoded.columns:
                                encoded[col] = 0
                        encoded = encoded[expected_cols]
                    
                    feature_df = pd.concat([feature_df, encoded], axis=1)
                    feature_df.drop(cat_col, axis=1, inplace=True)
        
        # Select final feature columns
        numeric_features = [col for col in core_features + derived_features 
                          if col in feature_df.columns and col not in categorical_features]
        
        encoded_features = []
        for cat_col in categorical_features:
            if cat_col in self.categorical_encoders:
                encoded_features.extend(self.categorical_encoders[cat_col])
        
        binary_features = ['high_ltv', 'high_dti', 'low_credit']
        
        all_features = numeric_features + encoded_features + binary_features
        
        if is_training:
            # Store feature columns for consistent ordering
            self.feature_columns = [col for col in all_features if col in feature_df.columns]
        
        # Ensure all expected features are present
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            
            X = feature_df[self.feature_columns]
        else:
            available_features = [col for col in all_features if col in feature_df.columns]
            X = feature_df[available_features]
        
        # Handle target column - only extract if it exists and is not None
        y = None
        if target_column and target_column in feature_df.columns:
            y = feature_df[target_column]
        
        self.logger.info(f"Prepared {len(X.columns)} features for modeling")
        self.logger.debug(f"Feature columns: {X.columns.tolist()}")
        
        return X, y
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using configured scaling method
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of scaled (X_train, X_test)
        """
        scaling_config = self.config.get('features.scaling', {})
        method = scaling_config.get('method', 'standard')
        features_to_scale = scaling_config.get('features_to_scale', [])
        
        # Filter features that exist in the data
        features_to_scale = [col for col in features_to_scale if col in X_train.columns]
        
        if not features_to_scale:
            self.logger.info("No features to scale")
            return X_train, X_test
        
        # Initialize scaler
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        self.logger.info(f"Scaling features using {method} scaler")
        
        # Create copies to avoid modifying original data
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Fit and transform training data
        X_train_scaled[features_to_scale] = self.scaler.fit_transform(X_train[features_to_scale])
        
        # Transform test data
        X_test_scaled[features_to_scale] = self.scaler.transform(X_test[features_to_scale])
        
        return X_train_scaled, X_test_scaled
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        data_config = self.config.get('data', {})
        test_size = data_config.get('test_size', 0.2)
        random_state = data_config.get('random_state', 42)
        stratify_column = data_config.get('stratify_column')
        
        stratify = y if stratify_column == y.name else None
        
        self.logger.info(f"Splitting data with test_size={test_size}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        self.logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, save_path: str):
        """
        Save the fitted preprocessor (scaler, feature columns, and categorical encoders)
        
        Args:
            save_path: Path to save the preprocessor
        """
        preprocessor_data = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'categorical_encoders': self.categorical_encoders,
            'categorical_categories': self.categorical_categories
        }
        
        joblib.dump(preprocessor_data, save_path)
        self.logger.info(f"Preprocessor saved to: {save_path}")
    
    def load_preprocessor(self, load_path: str):
        """
        Load a fitted preprocessor
        
        Args:
            load_path: Path to load the preprocessor from
        """
        preprocessor_data = joblib.load(load_path)
        self.scaler = preprocessor_data['scaler']
        self.feature_columns = preprocessor_data['feature_columns']
        self.categorical_encoders = preprocessor_data.get('categorical_encoders', {})
        self.categorical_categories = preprocessor_data.get('categorical_categories', {})
        self.logger.info(f"Preprocessor loaded from: {load_path}")
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor
        
        Args:
            df: New data to transform
            
        Returns:
            Transformed DataFrame ready for prediction
        """
        # Create features
        df_features = self.create_features(df)
        
        # Prepare features using prediction mode (is_training=False)
        X, _ = self.prepare_features(df_features, target_column=None, is_training=False)
        
        # Scale if scaler is available
        if self.scaler is not None:
            scaling_config = self.config.get('features.scaling', {})
            features_to_scale = scaling_config.get('features_to_scale', [])
            features_to_scale = [col for col in features_to_scale if col in X.columns]
            
            if features_to_scale:
                X_scaled = X.copy()
                X_scaled[features_to_scale] = self.scaler.transform(X[features_to_scale])
                X = X_scaled
        
        return X