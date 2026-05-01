"""
Streamlit Dashboard for Loan Outcome Prediction
Interactive web interface for loan application processing
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent))

from prediction_service import get_prediction_service
from config_manager import get_config

# Configure page
st.set_page_config(
    page_title="Loan Outcome Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        border-left-color: #ff4444 !important;
    }
    .risk-medium {
        border-left-color: #ffaa00 !important;
    }
    .risk-low {
        border-left-color: #00aa44 !important;
    }
    .prediction-approved {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .prediction-denied {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)


class LoanPredictionDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        """Initialize dashboard"""
        self.config = get_config()
        self.prediction_service = None
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize prediction service"""
        try:
            self.prediction_service = get_prediction_service()
            if not self.prediction_service.is_loaded:
                st.error("⚠️ Models not loaded. Please run the training pipeline first.")
                st.stop()
        except Exception as e:
            st.error(f"❌ Error initializing prediction service: {e}")
            st.stop()
    
    def run(self):
        """Run the dashboard"""
        # Header
        st.title("🏦 Loan Outcome Prediction System")
        st.markdown("**Production-grade AI system for loan application processing**")
        
        # Sidebar for navigation
        with st.sidebar:
            st.header("Navigation")
            page = st.selectbox(
                "Select Page",
                ["Loan Prediction", "Model Information", "Health Check", "About"]
            )
        
        # Route to selected page
        if page == "Loan Prediction":
            self._loan_prediction_page()
        elif page == "Model Information":
            self._model_info_page()
        elif page == "Health Check":
            self._health_check_page()
        elif page == "About":
            self._about_page()
    
    def _loan_prediction_page(self):
        """Main loan prediction interface"""
        st.header("Loan Application Prediction")
        
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Loan Application Details")
            
            # Input form
            with st.form("loan_application"):
                # Personal Information
                st.markdown("**Applicant Information**")
                income = st.number_input(
                    "Annual Income ($k)",
                    min_value=10.0,
                    max_value=1000.0,
                    value=75.0,
                    step=1.0,
                    help="Annual gross income in thousands of dollars"
                )
                
                credit_score = st.number_input(
                    "Credit Score",
                    min_value=300,
                    max_value=850,
                    value=720,
                    step=1,
                    help="FICO credit score (300-850)"
                )
                
                debt_to_income_ratio = st.number_input(
                    "Debt-to-Income Ratio (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=28.0,
                    step=0.1,
                    help="Monthly debt payments as percentage of gross monthly income"
                )
                
                st.markdown("**Loan Details**")
                loan_amount = st.number_input(
                    "Loan Amount ($)",
                    min_value=10000,
                    max_value=5000000,
                    value=300000,
                    step=1000,
                    help="Requested loan amount in dollars"
                )
                
                property_value = st.number_input(
                    "Property Value ($)",
                    min_value=15000,
                    max_value=10000000,
                    value=375000,
                    step=1000,
                    help="Estimated property value in dollars"
                )
                
                # Calculate LTV automatically
                ltv_ratio = (loan_amount / property_value * 100) if property_value > 0 else 0
                st.metric("Calculated Loan-to-Value Ratio", f"{ltv_ratio:.1f}%")
                
                # Submit button
                submit_button = st.form_submit_button("🔍 Predict Loan Outcome", use_container_width=True)
        
        with col2:
            st.subheader("Prediction Results")
            
            if submit_button:
                # Prepare loan data
                loan_data = {
                    'income': float(income),
                    'loan_amount': int(loan_amount),
                    'property_value': int(property_value),
                    'credit_score': int(credit_score),
                    'debt_to_income_ratio': float(debt_to_income_ratio),
                    'loan_to_value_ratio': float(ltv_ratio)
                }
                
                try:
                    # Make prediction
                    with st.spinner("Analyzing loan application..."):
                        results = self.prediction_service.predict_loan_outcome(loan_data)
                    
                    # Display results
                    self._display_prediction_results(results, loan_data)
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {e}")
            
            else:
                st.info("👆 Fill out the loan application form and click 'Predict Loan Outcome' to get results.")
    
    def _display_prediction_results(self, results: Dict[str, Any], loan_data: Dict[str, Any]):
        """Display prediction results"""
        outcome = results['outcome']
        confidence = results['confidence']
        
        # Main prediction result
        if outcome == "Approved":
            st.markdown(f"""
            <div class="prediction-approved">
                <h3>✅ LOAN APPROVED</h3>
                <p>Confidence: {confidence['approved']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            denial_reason = results.get('denial_reason', {})
            st.markdown(f"""
            <div class="prediction-denied">
                <h3>❌ LOAN DENIED</h3>
                <p>Confidence: {confidence['denied']:.1%}</p>
                <p><strong>Primary Reason:</strong> {denial_reason.get('description', 'Not specified')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence visualization
        st.subheader("Prediction Confidence")
        
        confidence_df = pd.DataFrame({
            'Outcome': ['Approved', 'Denied'],
            'Probability': [confidence['approved'], confidence['denied']]
        })
        
        # Simple bar chart using streamlit
        st.subheader("Prediction Probabilities")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="🟢 Approval Probability",
                value=f"{confidence['approved']:.1%}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="🔴 Denial Probability", 
                value=f"{confidence['denied']:.1%}",
                delta=None
            )
        
        # Simple bar chart
        st.bar_chart(confidence_df.set_index('Outcome'))
        
        # Risk factor analysis
        st.subheader("Risk Factor Analysis")
        risk_factors = results.get('risk_assessment', {})
        
        if risk_factors:
            risk_cols = st.columns(2)
            
            for i, (factor, assessment) in enumerate(risk_factors.items()):
                col = risk_cols[i % 2]
                
                with col:
                    color_class = f"risk-{assessment['level'].lower().replace(' ', '-')}"
                    st.markdown(f"""
                    <div class="metric-card {color_class}">
                        <h4>{factor.replace('_', ' ').title()}</h4>
                        <p><strong>Risk Level:</strong> {assessment['level']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Loan summary
        st.subheader("Application Summary")
        
        summary_cols = st.columns(3)
        
        with summary_cols[0]:
            st.metric("Loan Amount", f"${loan_data['loan_amount']:,.0f}")
            st.metric("Annual Income", f"${loan_data['income']:.0f}k")
        
        with summary_cols[1]:
            st.metric("Property Value", f"${loan_data['property_value']:,.0f}")
            st.metric("Credit Score", f"{loan_data['credit_score']}")
        
        with summary_cols[2]:
            st.metric("LTV Ratio", f"{loan_data['loan_to_value_ratio']:.1f}%")
            st.metric("DTI Ratio", f"{loan_data['debt_to_income_ratio']:.1f}%")
    
    def _model_info_page(self):
        """Display model information"""
        st.header("Model Information")
        
        # Get model info
        model_info = self.prediction_service.get_model_info()
        
        # Model status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Status")
            status_color = "🟢" if model_info['models_loaded'] else "🔴"
            st.write(f"{status_color} Models Loaded: {model_info['models_loaded']}")
            st.write(f"📁 Model Directory: {model_info['model_directory']}")
        
        with col2:
            st.subheader("Available Models")
            for model in model_info['available_models']:
                st.write(f"• {model['type'].replace('_', ' ').title()}: {model['algorithm']}")
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = self.prediction_service.get_feature_importance()
        
        if feature_importance:
            # Convert to DataFrame for visualization
            importance_df = pd.DataFrame(
                list(feature_importance.items())[:15],  # Top 15 features
                columns=['Feature', 'Importance']
            )
            
            # Simple horizontal bar chart using streamlit
            st.bar_chart(
                importance_df.set_index('Feature'),
                height=500
            )
            
            # Feature importance table
            st.subheader("Feature Importance Table")
            st.dataframe(importance_df, use_container_width=True)
        else:
            st.info("Feature importance information not available.")
    
    def _health_check_page(self):
        """Display system health check"""
        st.header("System Health Check")
        
        # Perform health check
        with st.spinner("Performing health check..."):
            health_results = self.prediction_service.health_check()
        
        # Overall status
        status = health_results['status']
        status_color = "🟢" if status == 'healthy' else "🔴"
        st.markdown(f"## {status_color} System Status: {status.upper()}")
        st.write(f"**Last Check:** {health_results['timestamp']}")
        
        # Individual checks
        st.subheader("Component Health")
        
        for check_name, check_result in health_results['checks'].items():
            status_icon = "✅" if check_result['status'] == 'pass' else "❌"
            
            with st.expander(f"{status_icon} {check_name.replace('_', ' ').title()}"):
                st.write(f"**Status:** {check_result['status']}")
                st.write(f"**Details:** {check_result['details']}")
        
        # Refresh button
        if st.button("🔄 Refresh Health Check"):
            st.experimental_rerun()
    
    def _about_page(self):
        """Display about information"""
        st.header("About Loan Prediction System")
        
        st.markdown("""
        ## Overview
        This is a production-grade machine learning system for automated loan outcome prediction.
        The system uses advanced ML algorithms to assess loan applications and provide 
        real-time approval/denial decisions with detailed reasoning.
        
        ## Features
        - **Real-time Predictions**: Instant loan outcome predictions
        - **Risk Assessment**: Detailed analysis of risk factors
        - **Denial Reasoning**: Specific reasons for loan denials
        - **Model Monitoring**: Health checks and performance monitoring
        - **Configurable**: Fully configurable model parameters
        
        ## Technology Stack
        - **Machine Learning**: Scikit-learn, XGBoost, LightGBM
        - **Web Framework**: Streamlit
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Plotly
        - **Configuration**: YAML-based configuration
        - **Logging**: Comprehensive logging system
        
        ## Model Information
        The system uses ensemble learning with multiple algorithms:
        - Random Forest Classifier
        - XGBoost Classifier (optional)
        - LightGBM Classifier (optional)
        - Logistic Regression
        
        ## Data Features
        The model considers multiple factors:
        - Credit score and history
        - Debt-to-income ratio
        - Loan-to-value ratio
        - Income level and stability
        - Property value and type
        - Historical loan performance
        
        ## Compliance
        - Fair lending practices
        - HMDA compliance for denial reasons
        - Transparent decision making
        - Audit trail and logging
        """)
        
        # System configuration
        st.subheader("System Configuration")
        config_info = {
            "Model Algorithm": self.config.get('loan_outcome_model.algorithm', 'Not specified'),
            "Feature Scaling": self.config.get('features.scaling.method', 'Not specified'),
            "Cross Validation": self.config.get('training.cross_validation.enabled', False),
            "Model Directory": self.config.get('persistence.model_dir', './models')
        }
        
        for key, value in config_info.items():
            st.write(f"**{key}:** {value}")


def main():
    """Main dashboard entry point"""
    try:
        dashboard = LoanPredictionDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        st.stop()


if __name__ == "__main__":
    main()