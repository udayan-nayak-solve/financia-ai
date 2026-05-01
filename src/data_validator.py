#!/usr/bin/env python3
"""
Comprehensive Data Validation and Sanitization Module

This module provides robust validation and sanitization for all data inputs
to ensure data quality and prevent issues with AI model training and predictions.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
from pathlib import Path

# Note: Logging is configured in the main pipeline module
logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validation and sanitization"""
    
    def __init__(self):
        """Initialize validator with validation rules"""
        
        # Define validation rules for each data type
        self.validation_rules = {
            'hmda': {
                'loan_amount': {'min': 1000, 'max': 10000000, 'required': True, 'type': 'numeric'},
                'applicant_income': {'min': 0, 'max': 10000000, 'required': False, 'type': 'numeric'},
                'income': {'min': 0, 'max': 10000000, 'required': False, 'type': 'numeric'},  # Alternative column name
                'action_taken': {'min': 1, 'max': 8, 'required': True, 'type': 'integer'},
                'census_tract': {'required': True, 'type': 'string', 'min_length': 3},
                'loan_type': {'min': 1, 'max': 4, 'required': False, 'type': 'integer'},
                'loan_purpose': {'min': 1, 'max': 32, 'required': False, 'type': 'integer'},
                'occupancy_type': {'min': 1, 'max': 3, 'required': False, 'type': 'integer'},
                'property_value': {'min': 1000, 'max': 50000000, 'required': False, 'type': 'numeric'},
                'debt_to_income_ratio': {'min': 0, 'max': 100, 'required': False, 'type': 'numeric'},
                'loan_to_value_ratio': {'min': 0, 'max': 120, 'required': False, 'type': 'numeric'},
                'applicant_credit_score_type': {'min': 300, 'max': 850, 'required': False, 'type': 'numeric'}
            },
            'gse': {
                'original_loan_amount': {'min': 1000, 'max': 10000000, 'required': True, 'type': 'numeric'},
                'original_interest_rate': {'min': 0.1, 'max': 25.0, 'required': True, 'type': 'numeric'},
                'ltv_ratio': {'min': 1, 'max': 120, 'required': False, 'type': 'numeric'},
                'dti_ratio': {'min': 0, 'max': 100, 'required': False, 'type': 'numeric'},
                'borrower_credit_score': {'min': 300, 'max': 850, 'required': False, 'type': 'numeric'},
                'census_tract': {'required': True, 'type': 'string', 'min_length': 3},
                'loan_term': {'min': 1, 'max': 50, 'required': False, 'type': 'integer'}
            },
            'census': {
                'census_tract': {'required': True, 'type': 'string', 'min_length': 3},
                'total_population': {'min': 0, 'max': 100000, 'required': False, 'type': 'integer'},
                'median_household_income': {'min': 0, 'max': 1000000, 'required': False, 'type': 'numeric'},
                'median_age': {'min': 0, 'max': 100, 'required': False, 'type': 'numeric'},
                'unemployment_rate': {'min': 0, 'max': 50, 'required': False, 'type': 'numeric'},
                'poverty_rate': {'min': 0, 'max': 100, 'required': False, 'type': 'numeric'},
                'education_bachelor_pct': {'min': 0, 'max': 100, 'required': False, 'type': 'numeric'},
                'homeownership_rate': {'min': 0, 'max': 100, 'required': False, 'type': 'numeric'}
            },
            'hpi': {
                'census_tract': {'required': True, 'type': 'string', 'min_length': 3},
                'hpi_value': {'min': 1, 'max': 10000, 'required': True, 'type': 'numeric'},
                'year': {'min': 1990, 'max': 2030, 'required': True, 'type': 'integer'},
                'quarter': {'min': 1, 'max': 4, 'required': False, 'type': 'integer'},
                'month': {'min': 1, 'max': 12, 'required': False, 'type': 'integer'}
            },
            'unemployment': {
                'county_code': {'required': True, 'type': 'string', 'min_length': 2},
                'unemployment_rate': {'min': 0, 'max': 50, 'required': True, 'type': 'numeric'},
                'year': {'min': 1990, 'max': 2030, 'required': True, 'type': 'integer'},
                'month': {'min': 1, 'max': 12, 'required': False, 'type': 'integer'}
            },
            'predictions': {
                'opportunity_score': {'min': 0, 'max': 100, 'required': True, 'type': 'numeric'},
                'predicted_opportunity_score': {'min': 0, 'max': 100, 'required': True, 'type': 'numeric'},
                'loan_volume': {'min': 0, 'max': 100000, 'required': False, 'type': 'integer'},
                'approval_rate': {'min': 0, 'max': 1, 'required': False, 'type': 'numeric'},
                'denial_rate': {'min': 0, 'max': 1, 'required': False, 'type': 'numeric'},
                'avg_loan_amount': {'min': 0, 'max': 10000000, 'required': False, 'type': 'numeric'}
            }
        }
        
        # Define loan prediction specific validation
        self.loan_prediction_rules = {
            'loan_amount': {'min': 1000, 'max': 5000000, 'required': True, 'type': 'numeric'},
            'income': {'min': 1000, 'max': 2000000, 'required': True, 'type': 'numeric'},
            'property_value': {'min': 10000, 'max': 20000000, 'required': True, 'type': 'numeric'},
            'credit_score': {'min': 300, 'max': 850, 'required': True, 'type': 'numeric'},
            'debt_to_income_ratio': {'min': 0, 'max': 100, 'required': True, 'type': 'numeric'},
            'loan_to_value_ratio': {'min': 1, 'max': 120, 'required': True, 'type': 'numeric'},
            'employment_years': {'min': 0, 'max': 50, 'required': False, 'type': 'numeric'},
            'loan_term': {'min': 1, 'max': 50, 'required': False, 'type': 'integer'},
            'loan_type': {'min': 1, 'max': 4, 'required': False, 'type': 'integer'},
            'loan_purpose': {'min': 1, 'max': 32, 'required': False, 'type': 'integer'},
            'occupancy_type': {'min': 1, 'max': 3, 'required': False, 'type': 'integer'}
        }
    
    def validate_dataframe(self, df: pd.DataFrame, data_type: str, 
                          strict: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate and sanitize a DataFrame according to data type rules
        
        Args:
            df: DataFrame to validate
            data_type: Type of data (hmda, gse, census, hpi, unemployment, predictions)
            strict: If True, reject rows with any validation errors. If False, sanitize values
            
        Returns:
            Tuple of (sanitized_df, validation_report)
        """
        if df.empty:
            return df, {'status': 'empty', 'errors': [], 'warnings': [], 'rows_processed': 0}
        
        logger.info(f"Validating {data_type} data: {len(df)} rows")
        
        # Get validation rules for this data type
        rules = self.validation_rules.get(data_type, {})
        if not rules:
            logger.warning(f"No validation rules defined for data type: {data_type}")
            return df, {'status': 'no_rules', 'errors': [], 'warnings': [], 'rows_processed': len(df)}
        
        df_clean = df.copy()
        validation_report = {
            'status': 'success',
            'errors': [],
            'warnings': [],
            'rows_processed': len(df),
            'rows_removed': 0,
            'values_sanitized': 0
        }
        
        # Track rows to remove (for strict mode)
        rows_to_remove = set()
        
        # Validate each column
        for column, rule in rules.items():
            if column not in df_clean.columns:
                if rule.get('required', False):
                    validation_report['errors'].append(f"Required column '{column}' missing")
                    continue
                else:
                    validation_report['warnings'].append(f"Optional column '{column}' missing")
                    continue
            
            # Validate and sanitize column
            df_clean, column_errors, column_warnings, removed_rows, sanitized_count = self._validate_column(
                df_clean, column, rule, strict
            )
            
            validation_report['errors'].extend(column_errors)
            validation_report['warnings'].extend(column_warnings)
            validation_report['values_sanitized'] += sanitized_count
            rows_to_remove.update(removed_rows)
        
        # Remove invalid rows if in strict mode
        if strict and rows_to_remove:
            df_clean = df_clean.drop(index=list(rows_to_remove))
            validation_report['rows_removed'] = len(rows_to_remove)
        
        # Additional data-specific validation
        df_clean, additional_report = self._additional_validation(df_clean, data_type)
        validation_report['errors'].extend(additional_report.get('errors', []))
        validation_report['warnings'].extend(additional_report.get('warnings', []))
        
        # Final status
        if validation_report['errors']:
            validation_report['status'] = 'errors'
        elif validation_report['warnings']:
            validation_report['status'] = 'warnings'
        
        logger.info(f"Validation complete: {len(df_clean)} rows remaining, "
                   f"{validation_report['values_sanitized']} values sanitized")
        
        return df_clean, validation_report
    
    def _validate_column(self, df: pd.DataFrame, column: str, rule: Dict, 
                        strict: bool) -> Tuple[pd.DataFrame, List, List, set, int]:
        """Validate and sanitize a single column"""
        
        errors = []
        warnings = []
        rows_to_remove = set()
        sanitized_count = 0
        
        # Handle missing values
        null_mask = df[column].isnull()
        if null_mask.any():
            null_count = null_mask.sum()
            if rule.get('required', False):
                if strict:
                    rows_to_remove.update(df.index[null_mask].tolist())
                    errors.append(f"Column '{column}': {null_count} required values are null")
                else:
                    # Fill with default values
                    default_val = self._get_default_value(rule)
                    df[column] = df[column].fillna(default_val)
                    sanitized_count += null_count
                    warnings.append(f"Column '{column}': {null_count} null values filled with {default_val}")
            else:
                warnings.append(f"Column '{column}': {null_count} null values in optional field")
        
        # Type conversion and validation
        if rule['type'] == 'numeric':
            df, type_errors, type_sanitized = self._validate_numeric_column(df, column, rule, strict)
        elif rule['type'] == 'integer':
            df, type_errors, type_sanitized = self._validate_integer_column(df, column, rule, strict)
        elif rule['type'] == 'string':
            df, type_errors, type_sanitized = self._validate_string_column(df, column, rule, strict)
        else:
            type_errors = []
            type_sanitized = 0
        
        errors.extend(type_errors)
        sanitized_count += type_sanitized
        
        return df, errors, warnings, rows_to_remove, sanitized_count
    
    def _validate_numeric_column(self, df: pd.DataFrame, column: str, rule: Dict, 
                                strict: bool) -> Tuple[pd.DataFrame, List, int]:
        """Validate numeric column"""
        errors = []
        sanitized_count = 0
        
        # Convert to numeric, coercing errors to NaN
        original_values = df[column].copy()
        df[column] = pd.to_numeric(df[column], errors='coerce')
        
        # Check for conversion errors
        conversion_errors = df[column].isnull() & original_values.notnull()
        if conversion_errors.any():
            error_count = conversion_errors.sum()
            errors.append(f"Column '{column}': {error_count} values could not be converted to numeric")
            if not strict:
                # Fill with default
                default_val = self._get_default_value(rule)
                df.loc[conversion_errors, column] = default_val
                sanitized_count += error_count
        
        # Check range constraints
        if 'min' in rule:
            below_min = df[column] < rule['min']
            if below_min.any():
                count = below_min.sum()
                if strict:
                    errors.append(f"Column '{column}': {count} values below minimum {rule['min']}")
                else:
                    df.loc[below_min, column] = rule['min']
                    sanitized_count += count
        
        if 'max' in rule:
            above_max = df[column] > rule['max']
            if above_max.any():
                count = above_max.sum()
                if strict:
                    errors.append(f"Column '{column}': {count} values above maximum {rule['max']}")
                else:
                    df.loc[above_max, column] = rule['max']
                    sanitized_count += count
        
        # Handle infinite values
        inf_mask = np.isinf(df[column])
        if inf_mask.any():
            count = inf_mask.sum()
            default_val = self._get_default_value(rule)
            df.loc[inf_mask, column] = default_val
            sanitized_count += count
            errors.append(f"Column '{column}': {count} infinite values replaced with {default_val}")
        
        return df, errors, sanitized_count
    
    def _validate_integer_column(self, df: pd.DataFrame, column: str, rule: Dict, 
                                strict: bool) -> Tuple[pd.DataFrame, List, int]:
        """Validate integer column"""
        errors = []
        sanitized_count = 0
        
        # First validate as numeric
        df, numeric_errors, numeric_sanitized = self._validate_numeric_column(df, column, rule, strict)
        errors.extend(numeric_errors)
        sanitized_count += numeric_sanitized
        
        # Convert to integer
        non_null_mask = df[column].notnull()
        if non_null_mask.any():
            # Round to nearest integer
            df.loc[non_null_mask, column] = df.loc[non_null_mask, column].round().astype(int)
        
        return df, errors, sanitized_count
    
    def _validate_string_column(self, df: pd.DataFrame, column: str, rule: Dict, 
                               strict: bool) -> Tuple[pd.DataFrame, List, int]:
        """Validate string column"""
        errors = []
        sanitized_count = 0
        
        # Convert to string
        df[column] = df[column].astype(str)
        
        # Replace 'nan' strings with appropriate defaults
        nan_mask = df[column].isin(['nan', 'NaN', 'None', ''])
        if nan_mask.any():
            if column in ['census_tract', 'county_code']:
                df.loc[nan_mask, column] = '000000'  # Default code
            else:
                df.loc[nan_mask, column] = 'Unknown'
            sanitized_count += nan_mask.sum()
        
        # Remove leading/trailing whitespace
        df[column] = df[column].str.strip()
        
        # Check minimum length
        if 'min_length' in rule:
            short_strings = df[column].str.len() < rule['min_length']
            if short_strings.any():
                count = short_strings.sum()
                if strict:
                    errors.append(f"Column '{column}': {count} values shorter than {rule['min_length']} characters")
                else:
                    # Pad with zeros for numeric codes, 'unknown' for others
                    if column in ['census_tract', 'county_code']:
                        df.loc[short_strings, column] = df.loc[short_strings, column].str.zfill(rule['min_length'])
                    else:
                        df.loc[short_strings, column] = 'Unknown'
                    sanitized_count += count
        
        return df, errors, sanitized_count
    
    def _get_default_value(self, rule: Dict) -> Any:
        """Get appropriate default value based on rule type"""
        if rule['type'] == 'numeric':
            if 'min' in rule and rule['min'] > 0:
                return float(rule['min'])
            return 0.0
        elif rule['type'] == 'integer':
            if 'min' in rule and rule['min'] > 0:
                return int(rule['min'])
            return 0
        elif rule['type'] == 'string':
            return 'Unknown'
        else:
            return None
    
    def _additional_validation(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, Dict]:
        """Perform additional data-specific validation"""
        report = {'errors': [], 'warnings': []}
        
        if data_type == 'hmda':
            # Validate loan-to-income ratios
            if 'loan_amount' in df.columns and 'applicant_income' in df.columns:
                income_col = 'applicant_income'
            elif 'loan_amount' in df.columns and 'income' in df.columns:
                income_col = 'income'
            else:
                income_col = None
            
            if income_col:
                # Calculate loan-to-income ratio and flag unrealistic values
                valid_income = df[income_col] > 0
                if valid_income.any():
                    lti_ratio = df.loc[valid_income, 'loan_amount'] / df.loc[valid_income, income_col]
                    unrealistic_lti = lti_ratio > 10  # More than 10x income
                    if unrealistic_lti.any():
                        count = unrealistic_lti.sum()
                        report['warnings'].append(f"HMDA: {count} loans have unrealistic loan-to-income ratios (>10x)")
            
            # Validate action_taken codes
            if 'action_taken' in df.columns:
                valid_actions = df['action_taken'].isin([1, 2, 3, 4, 5, 6, 7, 8])
                if not valid_actions.all():
                    invalid_count = (~valid_actions).sum()
                    report['errors'].append(f"HMDA: {invalid_count} invalid action_taken codes")
        
        elif data_type == 'gse':
            # Validate LTV + DTI combinations
            if 'ltv_ratio' in df.columns and 'dti_ratio' in df.columns:
                high_risk = (df['ltv_ratio'] > 95) & (df['dti_ratio'] > 45)
                if high_risk.any():
                    count = high_risk.sum()
                    report['warnings'].append(f"GSE: {count} loans have high risk profile (LTV>95% & DTI>45%)")
        
        elif data_type == 'predictions':
            # Validate opportunity scores are reasonable
            if 'opportunity_score' in df.columns:
                extreme_scores = (df['opportunity_score'] < 10) | (df['opportunity_score'] > 90)
                if extreme_scores.any():
                    count = extreme_scores.sum()
                    report['warnings'].append(f"Predictions: {count} extreme opportunity scores (<10 or >90)")
        
        return df, report
    
    def validate_loan_application(self, application_data: Dict) -> Tuple[Dict, Dict]:
        """
        Validate loan application data for prediction
        
        Args:
            application_data: Dictionary containing loan application data
            
        Returns:
            Tuple of (sanitized_data, validation_report)
        """
        logger.info("Validating loan application data")
        
        sanitized_data = application_data.copy()
        validation_report = {
            'status': 'success',
            'errors': [],
            'warnings': [],
            'values_sanitized': 0
        }
        
        # Validate each field
        for field, rule in self.loan_prediction_rules.items():
            if field not in sanitized_data:
                if rule.get('required', False):
                    validation_report['errors'].append(f"Required field '{field}' missing")
                    # Add default value
                    sanitized_data[field] = self._get_default_value(rule)
                    validation_report['values_sanitized'] += 1
                continue
            
            value = sanitized_data[field]
            
            # Type validation and conversion
            if rule['type'] == 'numeric':
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    validation_report['errors'].append(f"Field '{field}': Invalid numeric value")
                    value = self._get_default_value(rule)
                    validation_report['values_sanitized'] += 1
            elif rule['type'] == 'integer':
                try:
                    value = int(float(value))  # Handle string numbers
                except (ValueError, TypeError):
                    validation_report['errors'].append(f"Field '{field}': Invalid integer value")
                    value = self._get_default_value(rule)
                    validation_report['values_sanitized'] += 1
            
            # Range validation
            if 'min' in rule and value < rule['min']:
                validation_report['warnings'].append(f"Field '{field}': Value {value} below minimum {rule['min']}")
                value = rule['min']
                validation_report['values_sanitized'] += 1
            
            if 'max' in rule and value > rule['max']:
                validation_report['warnings'].append(f"Field '{field}': Value {value} above maximum {rule['max']}")
                value = rule['max']
                validation_report['values_sanitized'] += 1
            
            sanitized_data[field] = value
        
        # Additional loan application validation
        sanitized_data, additional_report = self._validate_loan_ratios(sanitized_data)
        validation_report['warnings'].extend(additional_report.get('warnings', []))
        validation_report['values_sanitized'] += additional_report.get('values_sanitized', 0)
        
        # Final status
        if validation_report['errors']:
            validation_report['status'] = 'errors'
        elif validation_report['warnings']:
            validation_report['status'] = 'warnings'
        
        logger.info(f"Loan application validation complete: {validation_report['status']}")
        return sanitized_data, validation_report
    
    def _validate_loan_ratios(self, data: Dict) -> Tuple[Dict, Dict]:
        """Validate loan application ratios and relationships"""
        report = {'warnings': [], 'values_sanitized': 0}
        
        # Calculate derived ratios if base values exist
        if 'loan_amount' in data and 'income' in data and data['income'] > 0:
            loan_to_income = data['loan_amount'] / data['income']
            if loan_to_income > 8:  # Extremely high loan-to-income
                report['warnings'].append(f"Very high loan-to-income ratio: {loan_to_income:.1f}x")
        
        if 'loan_amount' in data and 'property_value' in data and data['property_value'] > 0:
            ltv = (data['loan_amount'] / data['property_value']) * 100
            if 'loan_to_value_ratio' not in data:
                data['loan_to_value_ratio'] = ltv
                report['values_sanitized'] += 1
            elif abs(data['loan_to_value_ratio'] - ltv) > 5:  # Inconsistent LTV
                report['warnings'].append(f"Inconsistent LTV ratio provided vs calculated")
                data['loan_to_value_ratio'] = ltv  # Use calculated value
                report['values_sanitized'] += 1
        
        # Validate DTI ratio reasonableness
        if 'debt_to_income_ratio' in data:
            if data['debt_to_income_ratio'] > 50:
                report['warnings'].append(f"Very high DTI ratio: {data['debt_to_income_ratio']}%")
            elif data['debt_to_income_ratio'] < 5:
                report['warnings'].append(f"Unusually low DTI ratio: {data['debt_to_income_ratio']}%")
        
        return data, report
    
    def generate_validation_summary(self, validation_reports: List[Dict]) -> Dict:
        """Generate summary of validation results across multiple datasets"""
        
        summary = {
            'total_datasets': len(validation_reports),
            'successful': 0,
            'with_warnings': 0,
            'with_errors': 0,
            'total_rows_processed': 0,
            'total_values_sanitized': 0,
            'common_issues': {}
        }
        
        all_errors = []
        all_warnings = []
        
        for report in validation_reports:
            summary['total_rows_processed'] += report.get('rows_processed', 0)
            summary['total_values_sanitized'] += report.get('values_sanitized', 0)
            
            if report['status'] == 'success':
                summary['successful'] += 1
            elif report['status'] == 'warnings':
                summary['with_warnings'] += 1
            elif report['status'] == 'errors':
                summary['with_errors'] += 1
            
            all_errors.extend(report.get('errors', []))
            all_warnings.extend(report.get('warnings', []))
        
        # Identify common issues
        for error in all_errors:
            issue_type = error.split(':')[0] if ':' in error else error
            summary['common_issues'][issue_type] = summary['common_issues'].get(issue_type, 0) + 1
        
        summary['recommendations'] = self._generate_recommendations(summary, all_errors, all_warnings)
        
        return summary
    
    def _generate_recommendations(self, summary: Dict, errors: List, warnings: List) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        if summary['with_errors'] > 0:
            recommendations.append("❌ Critical data quality issues detected. Review and fix data sources.")
        
        if summary['total_values_sanitized'] > summary['total_rows_processed'] * 0.1:
            recommendations.append("⚠️ High sanitization rate (>10%). Consider improving data collection processes.")
        
        # Specific recommendations based on common issues
        missing_income_errors = [e for e in errors if 'income' in e.lower() and 'missing' in e.lower()]
        if missing_income_errors:
            recommendations.append("💰 Income data quality issues detected. Implement income validation at data source.")
        
        ratio_warnings = [w for w in warnings if 'ratio' in w.lower()]
        if ratio_warnings:
            recommendations.append("📊 Ratio inconsistencies found. Implement cross-field validation.")
        
        if not recommendations:
            recommendations.append("✅ Data quality looks good! Continue monitoring validation metrics.")
        
        return recommendations


def create_validation_report(validator: DataValidator, data_dict: Dict[str, pd.DataFrame]) -> None:
    """Create comprehensive validation report and save to file"""
    
    validation_reports = []
    
    # Validate each dataset
    for data_type, df in data_dict.items():
        if not df.empty:
            clean_df, report = validator.validate_dataframe(df, data_type, strict=False)
            validation_reports.append({
                'data_type': data_type,
                'original_rows': len(df),
                'clean_rows': len(clean_df),
                **report
            })
    
    # Generate summary
    summary = validator.generate_validation_summary(validation_reports)
    
    # Create report
    report_content = f"""
# Data Validation Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Datasets: {summary['total_datasets']}
- Successful: {summary['successful']}
- With Warnings: {summary['with_warnings']}
- With Errors: {summary['with_errors']}
- Total Rows Processed: {summary['total_rows_processed']:,}
- Total Values Sanitized: {summary['total_values_sanitized']:,}

## Recommendations
"""
    
    for rec in summary['recommendations']:
        report_content += f"- {rec}\n"
    
    report_content += "\n## Detailed Results\n"
    
    for report in validation_reports:
        report_content += f"""
### {report['data_type'].upper()} Data
- Status: {report['status']}
- Rows: {report['original_rows']} → {report['clean_rows']}
- Values Sanitized: {report['values_sanitized']}
- Errors: {len(report['errors'])}
- Warnings: {len(report['warnings'])}
"""
    
    # Save report
    report_path = Path("data/outputs/validation_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Validation report saved to {report_path}")


if __name__ == "__main__":
    # Test the validator
    validator = DataValidator()
    
    # Test loan application validation
    test_application = {
        'loan_amount': 300000,
        'income': 75000,
        'property_value': 400000,
        'credit_score': 720,
        'debt_to_income_ratio': 35,
        'loan_to_value_ratio': 75
    }
    
    clean_app, report = validator.validate_loan_application(test_application)
    print("Validation Report:", report)
    print("Clean Application:", clean_app)