#!/usr/bin/env python3
"""
Comprehensive Executive Dashboard System

Production-grade dashboard for lending opportunity platform providing:
1. Interactive visualizations for opportunity scores  
2. Market segment analysis and clustering
3. Loan outcome predictions and demos
4. Strategic insights and recommendations
5. Executive-level KPIs and metrics
6. Feature importance analysis
7. Economic indicators tracking
8. Risk assessment and management

Built with Streamlit for modern, interactive web interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
import json
import logging
from datetime import datetime
import warnings
import logging
import os
warnings.filterwarnings('ignore')
# Suppress specific plotly warnings from Streamlit internal usage
warnings.filterwarnings('ignore', message='.*keyword arguments have been deprecated.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='plotly.*')

# Suppress plotly deprecation messages at the logging level
logging.getLogger('plotly').setLevel(logging.ERROR)
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'

# Custom modules
from advanced_lending_platform import LendingConfig, DataProcessor, OpportunityScoreCalculator
from loan_outcome_predictor import LoanOutcomePredictor
from opportunity_forecaster import OpportunityForecaster
from hmda_temporal_forecaster import HMDAOpportunityForecaster
from market_segmenter import MarketSegmenter

    # Import for enhanced loan prediction and pickle compatibility
try:
    from enhanced_loan_predictor import EnhancedLoanPredictor
    from comprehensive_pipeline import EnhancedConfig
    ENHANCED_PREDICTOR_AVAILABLE = True
except ImportError as e:
    ENHANCED_PREDICTOR_AVAILABLE = False
    print(f"Enhanced predictor import failed: {e}")

# Import enhanced year-over-year analysis
try:
    from enhanced_yoy_dashboard import create_enhanced_yoy_dashboard
    ENHANCED_YOY_AVAILABLE = True
except ImportError as e:
    ENHANCED_YOY_AVAILABLE = False
    print(f"Enhanced YoY analysis import failed: {e}")# Configuration
st.set_page_config(
    page_title="Kansas Lending Opportunity Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border: none;
    }
    
    .metric-card h3 {
        color: white !important;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .metric-card .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    
    .metric-card .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.3rem;
    }
    
    .metric-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .metric-card-blue {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-card-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .metric-card-purple {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .info-box {
        background-color: #e8f4fd;
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .section-header {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all_data():
    """Load all data sources with caching"""
    config = LendingConfig()
    processor = DataProcessor(config)
    
    try:
        data_sources = processor.load_all_data()
        master_data = processor.create_master_dataset()
        
        # Calculate opportunity scores
        calculator = OpportunityScoreCalculator(config)
        scored_data = calculator.calculate_opportunity_score(master_data)
        
        return scored_data, config
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None


def load_market_segments_uncached(scored_data, config):
    """Load market segmentation analysis without caching"""
    try:
        from market_segmenter import perform_comprehensive_segmentation
        
        # Convert to fresh DataFrame to avoid caching issues
        data_copy = scored_data.copy()
        
        segmented_data, analysis_results = perform_comprehensive_segmentation(data_copy, config)
        return segmented_data, analysis_results
    except Exception as e:
        st.error(f"Error in market segmentation: {str(e)}")
        st.error(f"Details: {type(e).__name__}: {e}")
        return scored_data, {}



@st.cache_data
def load_forecasts(scored_data, config):
    """Load opportunity forecasts"""
    try:
        from opportunity_forecaster import create_comprehensive_forecasts
        forecasts, results = create_comprehensive_forecasts(scored_data, config)
        return forecasts, results
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")
        return None, {}


def load_temporal_forecasts(_config):
    """Load pre-computed temporal opportunity forecasts from saved files"""
    try:
        output_dir = Path("data/outputs")
        
        # Check for latest results files
        results_file = output_dir / "temporal_forecasting_results_latest.json"
        historical_file = output_dir / "historical_opportunity_scores_latest.json"
        predictions_file = output_dir / "future_predictions_latest.json"
        performance_file = output_dir / "model_performance_latest.json"
        
        required_files = [results_file, historical_file, predictions_file, performance_file]
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            st.error("⚠️ Pre-computed temporal forecasting results not found!")
            st.error(f"Missing files: {[f.name for f in missing_files]}")
            
            if st.button("🔄 Run Temporal Forecasting Pipeline Now"):
                with st.spinner("Running temporal forecasting pipeline... This may take a few minutes."):
                    from temporal_forecasting_pipeline import run_temporal_forecasting_pipeline
                    pipeline_results = run_temporal_forecasting_pipeline()
                    
                    if pipeline_results['success']:
                        st.success("✅ Temporal forecasting pipeline completed! Reloading page...")
                        st.experimental_rerun()
                    else:
                        st.error(f"Pipeline failed: {pipeline_results['error']}")
            
            st.info("💡 **To generate forecasting results, run:**")
            st.code("python3 src/temporal_forecasting_pipeline.py")
            return None, None
        
        # Load pre-computed results
        #st.info("📊 Loading pre-computed temporal forecasting results...")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        with open(historical_file, 'r') as f:
            historical_data = json.load(f)
        
        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)
        
        with open(performance_file, 'r') as f:
            performance_data = json.load(f)
        
        # Create a simple data structure instead of mock forecaster object
        forecaster_data = {
            'yearly_scores': {},
            'predictions': {}
        }
        
        # Convert historical data to simple dictionaries
        for year, year_data in historical_data['data_by_year'].items():
            forecaster_data['yearly_scores'][int(year)] = pd.DataFrame(year_data['data'])
        
        # Convert predictions data to simple dictionaries
        for year, year_data in predictions_data['data_by_year'].items():
            forecaster_data['predictions'][int(year)] = pd.DataFrame(year_data['data'])
        
        # Add timestamp info to results
        results['data_timestamp'] = historical_data.get('timestamp', 'Unknown')
        results['predictions_timestamp'] = predictions_data.get('timestamp', 'Unknown')
        
        #st.success(f"✅ Loaded pre-computed results (Generated: {results['data_timestamp'][:19]})")
        
        return forecaster_data, results
        
    except Exception as e:
        st.error(f"Error loading temporal forecasting results: {str(e)}")
        st.info("💡 Try running the temporal forecasting pipeline:")
        st.code("python3 src/temporal_forecasting_pipeline.py")
        return None, None
class SimpleConfig:
    """Simple config class for dashboard compatibility"""
    def __init__(self, base_config):
        self.BASE_DIR = base_config.BASE_DIR
        self.DATA_DIR = base_config.DATA_DIR
        self.balance_method = 'smote'
        self.test_size = 0.2
        self.cv_folds = 5

def load_loan_predictor():
    """Load best available loan outcome predictor with enhanced models from pipeline"""
    if not ENHANCED_PREDICTOR_AVAILABLE:
        st.error("Enhanced loan predictor not available. Please check imports.")
        return None
        
    try:
        import pickle
        
        config = LendingConfig()
        
        # Try to load enhanced models saved by the comprehensive pipeline
        enhanced_model_path = config.MODELS_DIR / 'enhanced_loan_models.pkl'
        
        if enhanced_model_path.exists():
            # Load the complete enhanced predictor with all models and metadata
            with open(enhanced_model_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            # Extract the enhanced predictor
            predictor = saved_data['enhanced_predictor']
            training_metadata = saved_data.get('training_metadata', {})
            
            # Ensure the predictor is marked as trained and has required attributes
            predictor.is_trained = True
            if hasattr(saved_data, 'feature_columns'):
                predictor.feature_columns = saved_data['feature_columns']
            elif hasattr(predictor, 'modeling_pipeline') and hasattr(predictor.modeling_pipeline, 'feature_columns'):
                predictor.feature_columns = predictor.modeling_pipeline.feature_columns
            
            # Ensure the modeling pipeline has the best model set
            if hasattr(predictor, 'modeling_pipeline') and hasattr(predictor.modeling_pipeline, 'models'):
                models = predictor.modeling_pipeline.models
                if models and (not hasattr(predictor.modeling_pipeline, 'best_model') or predictor.modeling_pipeline.best_model is None):
                    # Set the best model from the trained models
                    best_model_name = max(models.keys(), key=lambda k: models[k].get('test_f1', 0))
                    predictor.modeling_pipeline.best_model = models[best_model_name]['model']
                    predictor.modeling_pipeline.best_model_name = best_model_name
                    st.info(f"🎯 Set best model: {best_model_name}")
                
                # Set the trained state
                predictor.modeling_pipeline.is_trained = True
                if hasattr(predictor, 'is_trained'):
                    predictor.is_trained = True
            
            # Display model information
            training_date = training_metadata.get('training_date', 'Unknown')
            feature_count = training_metadata.get('feature_count', 0)
            denial_rate = training_metadata.get('denial_rate', 0)
            
            st.success(f"✅ Enhanced loan prediction model loaded")
            st.info(f"📊 Model Info: {feature_count} features, {denial_rate:.1%} denial rate, trained: {training_date[:10] if training_date != 'Unknown' else 'Unknown'}")
            
            return predictor
        
        else:
            st.error("❌ No trained enhanced models found - please run the comprehensive pipeline first")
            st.info("💡 Run: `python src/comprehensive_pipeline.py` to train models")
            return None
            
    except Exception as e:
        st.error(f"Error loading enhanced loan predictor: {str(e)}")
        st.error(f"Details: {type(e).__name__}: {e}")
        return None


def render_kpi_metrics(df):
    """Render key performance indicators with beautiful cards"""
    st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    total_tracts = len(df)
    avg_score = df['opportunity_score'].mean()
    high_opp = len(df[df['opportunity_level'].isin(['High'])])
    total_population = df['total_population'].sum()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card metric-card-blue">
            <h3>📍 Total Markets</h3>
            <div class="metric-value">{total_tracts:,}</div>
            <div class="metric-label">Census Tracts Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        score_color = "metric-card-green" if avg_score >= 60 else "metric-card-orange" if avg_score >= 45 else "metric-card-red"
        st.markdown(f"""
        <div class="metric-card {score_color}">
            <h3>🎯 Avg Opportunity Score</h3>
            <div class="metric-value">{avg_score:.1f}</div>
            <div class="metric-label">Out of 100 points</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        opportunity_pct = (high_opp / total_tracts * 100) if total_tracts > 0 else 0
        st.markdown(f"""
        <div class="metric-card metric-card-purple">
            <h3>🚀 High Opportunity</h3>
            <div class="metric-value">{high_opp:,}</div>
            <div class="metric-label">{opportunity_pct:.1f}% of all markets</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card metric-card-green">
            <h3>👥 Total Population</h3>
            <div class="metric-value">{total_population/1000000:.1f}M</div>
            <div class="metric-label">Market Population</div>
        </div>
        """, unsafe_allow_html=True)


def create_opportunity_overview(df):
    """Create comprehensive opportunity score overview section"""
    st.markdown('<div class="main-header">🏦 Kansas Lending Opportunity Platform - Overview</div>', unsafe_allow_html=True)
    
    # KPI Metrics first
    render_kpi_metrics(df)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Opportunity Score Distribution
        st.subheader("📊 Opportunity Score Distribution")
        
        fig = px.histogram(
            df, 
            x='opportunity_score',
            nbins=25,
            title='Distribution of Opportunity Scores Across Kansas Census Tracts',
            labels={'opportunity_score': 'Opportunity Score', 'count': 'Number of Census Tracts'},
            color_discrete_sequence=['#3498db'],
            opacity=0.8
        )
        
        # Add mean line
        mean_score = df['opportunity_score'].mean()
        fig.add_vline(
            x=mean_score, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Average: {mean_score:.1f}",
            annotation_position="top"
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Opportunity Categories Pie Chart
        st.subheader("🎯 Opportunity Categories")
        
        category_counts = df['opportunity_level'].value_counts()
        
        colors = {
            'High': '#27ae60',
            'Medium': '#f39c12',
            'Low': '#e74c3c'
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=category_counts.index,
            values=category_counts.values,
            marker=dict(colors=[colors.get(cat, '#95a5a6') for cat in category_counts.index]),
            hole=0.4,
            textinfo='label+percent',
            textfont_size=12
        )])
        
        fig.update_layout(
            title={
                'text': 'Market Opportunity Categories',
                'x': 0.5,
                'xanchor': 'center'
            },
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Opportunities Table
    st.subheader("🏆 Top 10 Opportunity Markets")
    
    top_markets = df.nlargest(10, 'opportunity_score')
    
    display_cols = ['census_tract', 'opportunity_score', 'opportunity_level', 
                   'market_accessibility', 'risk_factors', 'economic_indicators', 
                   'lending_activity', 'total_population', 'median_household_income']
    
    # Filter columns that exist in the dataframe
    available_cols = [col for col in display_cols if col in df.columns]
    display_df = top_markets[available_cols].copy()
    
    # Format for better display
    if 'median_household_income' in display_df.columns:
        display_df['median_household_income'] = display_df['median_household_income'].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Economic Indicators Analysis
    st.subheader("📈 Economic Indicators Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Unemployment vs Opportunity Score
        fig = px.scatter(
            df,
            x='unemployment_rate',
            y='opportunity_score',
            color='opportunity_level',
            title='Unemployment Rate vs Opportunity Score',
            labels={'unemployment_rate': 'Unemployment Rate (%)', 'opportunity_score': 'Opportunity Score'},
            color_discrete_map={'High': '#27ae60', 'Medium': '#f39c12', 'Low': '#e74c3c'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Income vs Opportunity Score
        fig = px.scatter(
            df,
            x='median_household_income',
            y='opportunity_score',
            color='opportunity_level',
            title='Household Income vs Opportunity Score',
            labels={'median_household_income': 'Median Household Income ($)', 'opportunity_score': 'Opportunity Score'},
            color_discrete_map={'High': '#27ae60', 'Medium': '#f39c12', 'Low': '#e74c3c'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Population vs Opportunity Score
        fig = px.scatter(
            df,
            x='total_population',
            y='opportunity_score',
            color='opportunity_level',
            title='Population vs Opportunity Score',
            labels={'total_population': 'Total Population', 'opportunity_score': 'Opportunity Score'},
            color_discrete_map={'High': '#27ae60', 'Medium': '#f39c12', 'Low': '#e74c3c'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


def create_component_analysis(df):
    """Create detailed component analysis view"""
    st.markdown('<div class="main-header">🔍 Component Analysis</div>', unsafe_allow_html=True)
    
    # Component score distributions
    component_cols = ['market_accessibility', 'risk_factors', 'economic_indicators', 'lending_activity']
    available_components = [col for col in component_cols if col in df.columns]
    
    if not available_components:
        st.warning("Component score data not available")
        return
    
    st.subheader("📊 Component Score Distributions")
    
    # Create subplots for component distributions
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=available_components,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    for i, component in enumerate(available_components):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        # Add histogram
        fig.add_trace(
            go.Histogram(x=df[component], name=component, nbinsx=20),
            row=row, col=col
        )
    
    fig.update_layout(height=600, showlegend=False, title_text="Component Score Distributions")
    st.plotly_chart(fig, use_container_width=True)
    
    # Component correlation analysis
    st.subheader("🔗 Component Correlation Matrix")
    
    correlation_data = df[available_components + ['opportunity_score']].corr()
    
    fig = px.imshow(
        correlation_data,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title='Component Correlation Matrix'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Component statistics summary
    st.subheader("📋 Component Statistics Summary")
    
    component_stats = df[available_components].describe().round(2)
    st.dataframe(component_stats, use_container_width=True)


def create_business_segments(df):
    """Create business-friendly market segments based on income and risk characteristics"""
    
    # Calculate percentiles for segmentation
    income_percentiles = df['median_household_income'].quantile([0.33, 0.67]) if 'median_household_income' in df.columns else [50000, 75000]
    risk_indicators = []
    
    # Create risk score based on available indicators
    if 'unemployment_rate' in df.columns:
        risk_indicators.append(df['unemployment_rate'])
    if 'denial_rate' in df.columns:
        risk_indicators.append(df['denial_rate'] * 100)  # Convert to percentage
    
    # If no risk indicators, use opportunity score inverse
    if not risk_indicators:
        risk_score = 100 - df['opportunity_score']  # Higher opportunity = lower risk
    else:
        # Combine risk indicators (normalize to 0-100 scale)
        risk_score = pd.concat(risk_indicators, axis=1).mean(axis=1) if len(risk_indicators) > 1 else risk_indicators[0]
    
    risk_percentiles = risk_score.quantile([0.33, 0.67])
    
    # Get income levels (use median_household_income if available, otherwise use approximation)
    if 'median_household_income' in df.columns:
        income = df['median_household_income']
    else:
        # Approximate income based on loan amounts and approval rates
        income = df['avg_loan_amount'] * 0.25 if 'avg_loan_amount' in df.columns else df['opportunity_score'] * 1000
    
    # Create segment labels
    def assign_segment(row):
        income_val = income.loc[row.name] if hasattr(income, 'loc') else income[row.name]
        risk_val = risk_score.loc[row.name] if hasattr(risk_score, 'loc') else risk_score[row.name]
        
        # Determine income level
        if income_val >= income_percentiles.iloc[1]:
            income_level = "High Income"
        elif income_val >= income_percentiles.iloc[0]:
            income_level = "Medium Income"
        else:
            income_level = "Lower Income"
        
        # Determine risk level (lower values = lower risk)
        if risk_val <= risk_percentiles.iloc[0]:
            risk_level = "Low Risk"
        elif risk_val <= risk_percentiles.iloc[1]:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
        
        return f"{income_level}, {risk_level}"
    
    # Apply segmentation
    df_with_segments = df.copy()
    df_with_segments['business_segment'] = df.apply(assign_segment, axis=1)
    
    return df_with_segments


def create_market_segmentation_view(segmented_data, analysis_results):
    """Enhanced market segmentation view with business-friendly segments"""
    
    st.markdown('<div class="main-header">🎯 Market Segmentation Analysis</div>', unsafe_allow_html=True)
    
    if segmented_data is None or segmented_data.empty:
        st.error("Market segmentation data not available.")
        return
    
    # Create business segments
    df_with_business_segments = create_business_segments(segmented_data)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_segments = len(df_with_business_segments['business_segment'].unique())
    largest_segment = df_with_business_segments['business_segment'].value_counts().index[0]
    largest_segment_size = df_with_business_segments['business_segment'].value_counts().iloc[0]
    avg_opportunity = df_with_business_segments['opportunity_score'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card metric-card-blue">
            <h3>📊 Total Segments</h3>
            <div class="metric-value">{total_segments}</div>
            <div class="metric-label">Distinct Market Segments</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card metric-card-green">
            <h3>🏆 Largest Segment</h3>
            <div class="metric-value">{largest_segment_size}</div>
            <div class="metric-label">{largest_segment}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        score_color = "metric-card-green" if avg_opportunity >= 60 else "metric-card-orange"
        st.markdown(f"""
        <div class="metric-card {score_color}">
            <h3>🎯 Avg Opportunity</h3>
            <div class="metric-value">{avg_opportunity:.1f}</div>
            <div class="metric-label">Across All Segments</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        high_opportunity_segments = len(df_with_business_segments[df_with_business_segments['opportunity_score'] >= 70]['business_segment'].unique())
        st.markdown(f"""
        <div class="metric-card metric-card-purple">
            <h3>� High-Opportunity</h3>
            <div class="metric-value">{high_opportunity_segments}</div>
            <div class="metric-label">Segments >70 Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main visualization section
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Enhanced segment distribution pie chart
        segment_counts = df_with_business_segments['business_segment'].value_counts()
        
        # Create color mapping based on segment characteristics
        colors = []
        for segment in segment_counts.index:
            if "High Income, Low Risk" in segment:
                colors.append("#27ae60")  # Green
            elif "High Income" in segment:
                colors.append("#3498db")  # Blue
            elif "Low Risk" in segment:
                colors.append("#2ecc71")  # Light green
            elif "High Risk" in segment:
                colors.append("#e74c3c")  # Red
            elif "Medium" in segment:
                colors.append("#f39c12")  # Orange
            else:
                colors.append("#95a5a6")  # Gray
        
        fig = go.Figure(data=[go.Pie(
            labels=segment_counts.index,
            values=segment_counts.values,
            marker=dict(colors=colors),
            hole=0.4,
            textinfo='label+percent+value',
            textfont_size=11,
            hovertemplate='<b>%{label}</b><br>Markets: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': 'Market Segment Distribution by Income & Risk Profile',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            height=500,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05,
                font=dict(size=10)
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Segment performance summary
        st.markdown("### 📊 Segment Performance")
        
        segment_stats = df_with_business_segments.groupby('business_segment').agg({
            'opportunity_score': ['mean', 'count'],
            'total_population': 'sum' if 'total_population' in df_with_business_segments.columns else 'count'
        }).round(1)
        
        # Sort by opportunity score
        segment_stats = segment_stats.sort_values(('opportunity_score', 'mean'), ascending=False)
        
        for segment in segment_stats.index:
            avg_score = segment_stats.loc[segment, ('opportunity_score', 'mean')]
            tract_count = int(segment_stats.loc[segment, ('opportunity_score', 'count')])
            
            # Determine performance color
            if avg_score >= 70:
                performance_icon = "🟢"
                performance_class = "success-box"
            elif avg_score >= 50:
                performance_icon = "🟡"
                performance_class = "info-box"
            else:
                performance_icon = "🔴"
                performance_class = "warning-box"
            
            st.markdown(f"""
            <div class="{performance_class}" style="margin: 0.5rem 0; padding: 0.5rem;">
                <strong>{performance_icon} {segment}</strong><br>
                <small>Score: {avg_score:.1f} | Markets: {tract_count}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed segment analysis
    st.markdown("---")
    st.markdown("### 🔍 Detailed Segment Analysis")
    
    # Create detailed analysis table
    detailed_stats = df_with_business_segments.groupby('business_segment').agg({
        'opportunity_score': ['mean', 'std', 'min', 'max', 'count'],
        'total_population': 'sum' if 'total_population' in df_with_business_segments.columns else 'count',
        'median_household_income': 'mean' if 'median_household_income' in df_with_business_segments.columns else lambda x: 0,
        'unemployment_rate': 'mean' if 'unemployment_rate' in df_with_business_segments.columns else lambda x: 0
    }).round(2)
    
    # Format the display
    display_data = []
    for segment in detailed_stats.index:
        avg_score = detailed_stats.loc[segment, ('opportunity_score', 'mean')]
        count = int(detailed_stats.loc[segment, ('opportunity_score', 'count')])
        population = int(detailed_stats.loc[segment, ('total_population', 'sum')]) if ('total_population', 'sum') in detailed_stats.columns else count
        
        # Determine strategic recommendation
        if "High Income, Low Risk" in segment:
            strategy = "🎯 Prime Target - Aggressive Expansion"
        elif "High Income" in segment and "Medium Risk" in segment:
            strategy = "📈 Growth Focus - Selective Expansion"
        elif "Medium Income, Low Risk" in segment:
            strategy = "🏦 Core Market - Steady Growth"
        elif "Low Risk" in segment:
            strategy = "✅ Stable Market - Maintain Presence"
        elif "High Risk" in segment:
            strategy = "⚠️ Caution - Limited Exposure"
        else:
            strategy = "📊 Standard Market - Balanced Approach"
        
        display_data.append({
            'Segment': segment,
            'Markets': count,
            'Population': f"{population:,}",
            'Avg Score': f"{avg_score:.1f}",
            'Strategy': strategy
        })
    
    # Sort by average score descending
    display_data = sorted(display_data, key=lambda x: float(x['Avg Score']), reverse=True)
    
    # Display as formatted table
    for data in display_data:
        with st.expander(f"**{data['Segment']}** - Score: {data['Avg Score']} | Markets: {data['Markets']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **📍 Market Size:** {data['Markets']} census tracts  
                **👥 Total Population:** {data['Population']}  
                **🎯 Average Opportunity Score:** {data['Avg Score']}
                """)
            with col2:
                st.markdown(f"""
                **💼 Strategic Recommendation:**  
                {data['Strategy']}
                """)
    
    # Original clustering analysis if available
    if analysis_results and 'segment_profiles' in analysis_results:
        st.markdown("---")
        st.markdown("### 🤖 AI Clustering Analysis")
        
        profiles = analysis_results['segment_profiles']
        
        if profiles:
            st.markdown("**Technical Segment Profiles:**")
            
            cols = st.columns(min(len(profiles), 3))
            for i, (segment_id, profile) in enumerate(profiles.items()):
                col_idx = i % 3
                with cols[col_idx]:
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>Cluster {segment_id}</h4>
                        <ul>
                            <li><strong>Size:</strong> {profile.get('size', 'N/A')} markets</li>
                            <li><strong>Avg Score:</strong> {profile.get('avg_opportunity_score', 0):.1f}</li>
                            <li><strong>Population:</strong> {profile.get('total_population', 0):,}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)


def create_tract_details_view(df, config=None):
    """Create detailed census tract analysis view with temporal opportunity scores"""
    st.markdown('<div class="main-header">📍 Census Tract Details</div>', unsafe_allow_html=True)
    
    # Tract selector
    st.subheader("🔍 Select Census Tract for Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Filter by opportunity level
        opportunity_filter = st.selectbox(
            "Filter by Opportunity Level:",
            ['All'] + list(df['opportunity_level'].unique())
        )
        
        if opportunity_filter != 'All':
            filtered_df = df[df['opportunity_level'] == opportunity_filter]
        else:
            filtered_df = df
    
    with col2:
        # Tract selection
        selected_tract = st.selectbox(
            "Select Census Tract:",
            filtered_df['census_tract'].tolist()
        )
    
    # Display selected tract details
    if selected_tract:
        tract_data = df[df['census_tract'] == selected_tract].iloc[0]
        
        st.subheader(f"📊 Census Tract {selected_tract} - Detailed Analysis")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card metric-card-blue">
                <h3>🎯 Opportunity Score</h3>
                <div class="metric-value">{tract_data['opportunity_score']:.1f}</div>
                <div class="metric-label">{tract_data['opportunity_level']} Opportunity</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card metric-card-green">
                <h3>👥 Population</h3>
                <div class="metric-value">{tract_data['total_population']:,.0f}</div>
                <div class="metric-label">Total Residents</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if 'median_household_income' in tract_data:
                st.markdown(f"""
                <div class="metric-card metric-card-purple">
                    <h3>💰 Median Income</h3>
                    <div class="metric-value">${tract_data['median_household_income']/1000:.0f}K</div>
                    <div class="metric-label">Household Income</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            if 'unemployment_rate' in tract_data:
                unemployment_color = "metric-card-green" if tract_data['unemployment_rate'] < 5 else "metric-card-orange"
                st.markdown(f"""
                <div class="metric-card {unemployment_color}">
                    <h3>📉 Unemployment</h3>
                    <div class="metric-value">{tract_data['unemployment_rate']:.1f}%</div>
                    <div class="metric-label">Current Rate</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Add temporal opportunity score analysis
        st.markdown("---")
        create_tract_temporal_analysis(selected_tract, config)


def create_tract_temporal_analysis(census_tract, config):
    """Create temporal opportunity score analysis for specific census tract"""
    st.subheader(f"📈 Temporal Opportunity Score Analysis - Tract {census_tract}")
    
    if config is None:
        st.warning("Configuration not available for temporal analysis")
        return
    
    # Load temporal forecasting data
    try:
        with st.spinner("Loading temporal data for selected tract..."):
            forecaster, results = load_temporal_forecasts(config)
        
        if forecaster is None or results is None:
            st.warning("⚠️ Temporal forecasting data not available.")
            st.info("💡 Run the temporal forecasting pipeline to see historical vs predicted trends.")
            return
        
        # Normalize census tract format for matching
        # Handle both integer format (20037957000) and float format (20037957000.0)
        target_tract_variants = [
            str(census_tract),                    # Original string
            str(float(census_tract)),             # Add .0 if integer
            str(int(float(census_tract))),        # Remove .0 if float
        ]
        
        #st.info(f"🔍 Searching for tract: {census_tract} (checking variants: {target_tract_variants})")
        
        # Collect opportunity scores for this specific tract across all years
        tract_timeline_data = []
        
        # Historical data (calculated scores)
        if 'yearly_scores' in forecaster:
            for year, year_data in forecaster['yearly_scores'].items():
                if 'census_tract' in year_data.columns:
                    # Try to find tract using any of the variants
                    tract_rows = pd.DataFrame()
                    for variant in target_tract_variants:
                        matches = year_data[year_data['census_tract'] == variant]
                        if not matches.empty:
                            tract_rows = matches
                            #st.success(f"✅ Found historical data for {year} using format: {variant}")
                            break
                    
                    for _, row in tract_rows.iterrows():
                        tract_timeline_data.append({
                            'year': int(year),
                            'opportunity_score': row['opportunity_score'],
                            'data_type': 'Historical (Calculated)',
                            'confidence': 100,  # Historical data has 100% confidence
                            'score_category': 'Calculated'
                        })
        
        # Future predictions (predicted scores)
        if 'predictions' in forecaster:
            for year, pred_data in forecaster['predictions'].items():
                if 'census_tract' in pred_data.columns:
                    # Try to find tract using any of the variants
                    tract_rows = pd.DataFrame()
                    for variant in target_tract_variants:
                        matches = pred_data[pred_data['census_tract'] == variant]
                        if not matches.empty:
                            tract_rows = matches
                            #st.success(f"✅ Found prediction data for {year} using format: {variant}")
                            break
                    
                    for _, row in tract_rows.iterrows():
                        tract_timeline_data.append({
                            'year': int(year),
                            'opportunity_score': row['predicted_opportunity_score'],
                            'data_type': 'Future (Predicted)',
                            'confidence': row.get('prediction_confidence', 0),
                            'score_category': 'Predicted'
                        })
        
        if not tract_timeline_data:
            st.error(f"❌ No temporal data found for Census Tract {census_tract}")
            
            # Debug information
            with st.expander("🔧 Debug Information"):
                st.write("**Available census tracts in historical data:**")
                if 'yearly_scores' in forecaster and forecaster['yearly_scores']:
                    first_year = list(forecaster['yearly_scores'].keys())[0]
                    first_year_data = forecaster['yearly_scores'][first_year]
                    available_tracts = first_year_data['census_tract'].unique()[:10]  # Show first 10
                    st.write(f"Sample tracts from {first_year}: {list(available_tracts)}")
                
                st.write("**Available census tracts in prediction data:**")
                if 'predictions' in forecaster and forecaster['predictions']:
                    first_pred_year = list(forecaster['predictions'].keys())[0]
                    first_pred_data = forecaster['predictions'][first_pred_year]
                    available_pred_tracts = first_pred_data['census_tract'].unique()[:10]  # Show first 10
                    st.write(f"Sample tracts from {first_pred_year}: {list(available_pred_tracts)}")
                
                st.write(f"**Searched for variants:** {target_tract_variants}")
            
            return
        
        # Create DataFrame for visualization
        timeline_df = pd.DataFrame(tract_timeline_data)
        timeline_df = timeline_df.sort_values('year')
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        historical_data = timeline_df[timeline_df['score_category'] == 'Calculated']
        predicted_data = timeline_df[timeline_df['score_category'] == 'Predicted']
        
        with col1:
            years_historical = len(historical_data['year'].unique())
            st.metric(
                label="Historical Years",
                value=f"{years_historical}",
                help="Years with calculated opportunity scores"
            )
        
        with col2:
            years_predicted = len(predicted_data['year'].unique())
            st.metric(
                label="Predicted Years", 
                value=f"{years_predicted}",
                help="Years with predicted opportunity scores"
            )
        
        with col3:
            if not historical_data.empty:
                avg_historical = historical_data['opportunity_score'].mean()
                st.metric(
                    label="Avg Historical Score",
                    value=f"{avg_historical:.1f}",
                    help="Average calculated opportunity score"
                )
        
        with col4:
            if not predicted_data.empty:
                avg_predicted = predicted_data['opportunity_score'].mean()
                if not historical_data.empty:
                    delta = avg_predicted - avg_historical
                    st.metric(
                        label="Avg Predicted Score",
                        value=f"{avg_predicted:.1f}",
                        delta=f"{delta:+.1f}",
                        help="Average predicted opportunity score vs historical"
                    )
                else:
                    st.metric(
                        label="Avg Predicted Score",
                        value=f"{avg_predicted:.1f}",
                        help="Average predicted opportunity score"
                    )
        
        # Create interactive timeline chart
        st.subheader("📊 Opportunity Score Timeline")
        
        # Main timeline chart with differentiated styling
        fig = px.line(
            timeline_df,
            x='year',
            y='opportunity_score',
            color='score_category',
            line_dash='score_category',
            markers=True,
            title=f"Opportunity Score Timeline - Census Tract {census_tract}",
            labels={
                'opportunity_score': 'Opportunity Score',
                'year': 'Year',
                'score_category': 'Data Type'
            },
            color_discrete_map={
                'Calculated': '#2E86AB',  # Blue for historical
                'Predicted': '#A23B72'   # Purple for predicted
            }
        )
        
        # Add confidence intervals for predictions
        if not predicted_data.empty:
            # Calculate confidence bands (±5 points based on confidence level)
            for _, row in predicted_data.iterrows():
                confidence_factor = row['confidence'] / 100
                margin = 5 * (1 - confidence_factor)  # Lower confidence = wider band
                
                fig.add_scatter(
                    x=[row['year'], row['year']],
                    y=[row['opportunity_score'] - margin, row['opportunity_score'] + margin],
                    mode='lines',
                    line=dict(color='rgba(162, 59, 114, 0.2)', width=0),
                    fill='tonexty',
                    fillcolor='rgba(162, 59, 114, 0.1)',
                    name=f'Confidence Band ({row["confidence"]:.0f}%)',
                    showlegend=False,
                    hovertemplate=f'Confidence: {row["confidence"]:.0f}%<extra></extra>'
                )
        
        # Add vertical line to separate historical from predicted
        current_year = 2024
        fig.add_vline(
            x=current_year + 0.5,
            line_dash="dash",
            line_color="gray",
            annotation_text="Historical | Predicted",
            annotation_position="top"
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Custom hover template
        fig.update_traces(
            hovertemplate='<b>Year:</b> %{x}<br>' +
                         '<b>Score:</b> %{y:.1f}<br>' +
                         '<b>Type:</b> %{fullData.name}<br>' +
                         '<extra></extra>'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed data table
        st.subheader("📋 Detailed Timeline Data")
        
        # Format data for display
        display_df = timeline_df.copy()
        display_df['Opportunity Score'] = display_df['opportunity_score'].round(1)
        display_df['Confidence (%)'] = display_df['confidence'].round(1)
        display_df['Year'] = display_df['year'].astype(int)
        display_df['Data Type'] = display_df['data_type']
        
        # Select columns for display
        display_columns = ['Year', 'Opportunity Score', 'Data Type', 'Confidence (%)']
        final_display_df = display_df[display_columns].sort_values('Year')
        
        # Color-code the rows based on data type
        def color_rows(row):
            if 'Historical' in str(row['Data Type']):
                return ['background-color: #E3F2FD'] * len(row)  # Light blue
            else:
                return ['background-color: #F3E5F5'] * len(row)  # Light purple
        
        st.dataframe(
            final_display_df.style.apply(color_rows, axis=1),
            use_container_width=True,
            height=400
        )
        
        # Trend analysis
        if len(timeline_df) > 1:
            st.subheader("📈 Trend Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Historical trend
                if len(historical_data) > 1:
                    historical_trend = historical_data['opportunity_score'].diff().mean()
                    trend_icon = "📈" if historical_trend > 0 else "📉" if historical_trend < 0 else "➡️"
                    trend_color = "green" if historical_trend > 0 else "red" if historical_trend < 0 else "gray"
                    
                    st.markdown(f"""
                    **Historical Trend ({historical_data['year'].min()}-{historical_data['year'].max()}):**
                    
                    {trend_icon} **{historical_trend:+.1f} points/year** average change
                    """)
                    
                    if abs(historical_trend) > 2:
                        st.markdown(f"<span style='color: {trend_color}'>**Significant trend detected**</span>", unsafe_allow_html=True)
                    elif abs(historical_trend) > 1:
                        st.markdown(f"<span style='color: orange'>**Moderate trend detected**</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("**Relatively stable scores**")
            
            with col2:
                # Predicted vs historical comparison
                if not historical_data.empty and not predicted_data.empty:
                    last_historical = historical_data['opportunity_score'].iloc[-1]
                    first_predicted = predicted_data['opportunity_score'].iloc[0]
                    change = first_predicted - last_historical
                    
                    change_icon = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    change_color = "green" if change > 0 else "red" if change < 0 else "gray"
                    
                    st.markdown(f"""
                    **Historical to Predicted Transition:**
                    
                    {change_icon} **{change:+.1f} points** expected change
                    """)
                    
                    if abs(change) > 5:
                        st.markdown(f"<span style='color: {change_color}'>**Major shift predicted**</span>", unsafe_allow_html=True)
                    elif abs(change) > 2:
                        st.markdown(f"<span style='color: orange'>**Moderate shift predicted**</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("**Stable transition expected**")
        
        # Opportunity Score Reasoning Section
        st.subheader("🧠 Opportunity Score Reasoning")
        
        # Find the most recent historical data with reasoning
        reasoning_data = None
        reasoning_year = None
        
        # Look for reasoning in historical data (most recent year)
        if 'yearly_scores' in forecaster and forecaster['yearly_scores']:
            for year in sorted(forecaster['yearly_scores'].keys(), reverse=True):
                year_data = forecaster['yearly_scores'][year]
                
                # Try to find this tract's data
                for variant in target_tract_variants:
                    matches = year_data[year_data['census_tract'] == variant]
                    if not matches.empty and 'reasoning' in matches.columns:
                        reasoning_entry = matches.iloc[0]['reasoning']
                        if reasoning_entry and isinstance(reasoning_entry, dict):
                            reasoning_data = reasoning_entry
                            reasoning_year = year
                            break
                if reasoning_data:
                    break
        
        if reasoning_data:
            st.markdown(f"**Analysis based on {reasoning_year} data:**")
            
            # Overall Assessment
            overall_score = reasoning_data.get('overall_score', 'N/A')
            overall_assessment = reasoning_data.get('assessment', 'No assessment available')
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(
                    label="Overall Score",
                    value=f"{overall_score}",
                    help="Weighted composite opportunity score"
                )
            
            with col2:
                st.info(f"**Assessment:** {overall_assessment}")
            
            # Component Breakdown
            st.markdown("### 📊 Component Analysis")
            
            components = reasoning_data.get('components', {})
            if components:
                # Create tabs for each component
                component_names = list(components.keys())
                component_tabs = st.tabs([name.replace('_', ' ').title() for name in component_names])
                
                for i, (component_name, component_tab) in enumerate(zip(component_names, component_tabs)):
                    with component_tab:
                        component_data = components[component_name]
                        
                        col1, col2, col3 = st.columns([1, 1, 3])
                        
                        with col1:
                            st.metric(
                                label="Score",
                                value=f"{component_data.get('score', 'N/A')}"
                            )
                        
                        with col2:
                            st.metric(
                                label="Weight",
                                value=component_data.get('weight', 'N/A')
                            )
                        
                        with col3:
                            st.markdown(f"**Reasoning:** {component_data.get('reasoning', 'No reasoning available')}")
            
            # Key Factors Summary
            st.markdown("### 🔑 Key Contributing Factors")
            key_factors = reasoning_data.get('key_factors', [])
            if key_factors:
                for factor in key_factors:
                    st.markdown(f"• {factor}")
            
            # Calculation Method
            calculation_method = reasoning_data.get('calculation_method', 'Standard weighted scoring methodology')
            with st.expander("🔬 Calculation Methodology"):
                st.markdown(f"**Method:** {calculation_method}")
                st.markdown("""
                **Scoring Components:**
                1. **Lending Activity (30%)** - Volume of loan applications in the area
                2. **Approval Rate (25%)** - Percentage of loans approved vs denied
                3. **Market Accessibility (20%)** - Based on loan amounts and property values
                4. **Economic Indicators (15%)** - Income levels and economic stability
                5. **Risk Factors (10%)** - Debt-to-income ratios and lending risk
                """)
        else:
            st.warning("🔍 Detailed reasoning data not available for this census tract.")
            st.info("💡 Run the updated temporal forecasting pipeline to generate reasoning data.")
    
        
    except Exception as e:
        st.error(f"Error loading temporal analysis: {str(e)}")
        st.info("💡 Ensure temporal forecasting pipeline has been run successfully.")


def create_loan_prediction_demo():
    """Create loan outcome prediction demonstration"""
    st.markdown('<div class="main-header">🤖 Loan Outcome Prediction Demo</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This tool demonstrates our AI-powered loan outcome prediction system. 
    Enter loan application details to get instant approval probability and risk assessment.
    """)
    
    # Load loan predictor
    predictor = load_loan_predictor()
    
    if predictor is None:
        st.error("Loan prediction models not available. Please run the training pipeline first.")
        return
    
    # Input form
    st.subheader("📝 Loan Application Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Basic loan details
        st.markdown("**Loan Information**")
        loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=2000000, value=250000, step=1000)
        loan_purpose = st.selectbox("Loan Purpose", [
            "Home purchase", "Refinancing", "Cash-out refinancing", "Home improvement"
        ])
        loan_type = st.selectbox("Loan Type", [
            "Conventional", "FHA", "VA", "USDA"
        ])
        property_value = st.number_input("Property Value ($)", min_value=1000, max_value=5000000, value=312500, step=1000)
    
    with col2:
        # Applicant details
        st.markdown("**Applicant Information**")
        income = st.number_input("Annual Income ($)", min_value=1000, max_value=1000000, value=75000, step=1000)
        credit_score = st.selectbox("Credit Score Range", [
            "800+", "760-799", "720-759", "680-719", "640-679", "600-639", "Below 600"
        ])
        ethnicity = st.selectbox("Ethnicity", [
            "Not Hispanic or Latino", "Hispanic or Latino", "Not provided"
        ])
        race = st.selectbox("Race", [
            "White", "Black or African American", "Asian", "American Indian or Alaska Native", 
            "Native Hawaiian or Other Pacific Islander", "Multiple races", "Not provided"
        ])
        sex = st.selectbox("Sex", ["Male", "Female", "Not provided"])
    
    with col3:
        # Financial ratios
        st.markdown("**Financial Ratios**")
        debt_to_income = st.slider("Debt-to-Income Ratio (%)", 0, 80, 28)
        loan_to_value = st.slider("Loan-to-Value Ratio (%)", 0, 100, 80)
        occupancy_type = st.selectbox("Occupancy Type", [
            "Owner-occupied", "Not owner-occupied", "Not applicable"
        ])
        lien_status = st.selectbox("Lien Status", [
            "First lien", "Subordinate lien"
        ])
    
    # Prediction button
    if st.button("🔍 Predict Loan Outcome", type="primary"):
        
        # Convert credit score to numeric value
        credit_score_map = {
            "800+": 820, "760-799": 780, "720-759": 740, "680-719": 700,
            "640-679": 660, "600-639": 620, "Below 600": 580
        }
        numeric_credit_score = credit_score_map.get(credit_score, 700)
        
        # Calculate derived financial metrics
        loan_to_income_ratio = loan_amount / max(income, 1)
        down_payment_ratio = 1 - (loan_amount / max(property_value, 1))
        
        # Prepare application data for enhanced model (including all required features)
        application_data = {
            # Most Important Features (HIGH Priority)
            'debt_to_income_ratio': float(debt_to_income),
            'loan_to_value_ratio': float(loan_to_value),
            'property_value': float(property_value),
            'loan_amount': float(loan_amount),
            'loan_purpose': {
                "Home purchase": 1, "Refinancing": 3, "Cash-out refinancing": 31, "Home improvement": 2
            }.get(loan_purpose, 1),
            'income': float(income),
            
            # Less Important Features (LOW Priority)
            'derived_ethnicity': ethnicity,
            'derived_race': race,
            'derived_sex': sex,
            
            # Additional Required Features
            'applicant_credit_score_type': float(numeric_credit_score),
            'loan_type': {
                "Conventional": 1, "FHA": 2, "VA": 3, "USDA": 4
            }.get(loan_type, 1),
            'occupancy_type': {
                "Owner-occupied": 1, "Not owner-occupied": 2, "Not applicable": 3
            }.get(occupancy_type, 1),
            'lien_status': {
                "First lien": 1, "Subordinate lien": 2
            }.get(lien_status, 1),
            
            # Derived Financial Metrics
            'loan_to_income_ratio': float(loan_to_income_ratio),
            'down_payment_ratio': float(down_payment_ratio),
            
            # Default Values for Model Compatibility
            'loan_term': 30,  # Default to 30-year
            'employment_years': 3.0,  # Default stable employment
            'derived_loan_product_type': 'Conventional:First Lien',
            'derived_dwelling_category': 'Single Family (1-4 Units):Site-Built'
        }
        
        try:
            # Get prediction from enhanced predictor
            prediction_result = predictor.predict_loan_outcome(application_data)
            
            st.subheader("🎯 Enhanced AI Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Approval probability
                approval_prob = prediction_result.get('approval_probability', 0) * 100
                prob_color = "metric-card-green" if approval_prob >= 70 else "metric-card-orange" if approval_prob >= 40 else "metric-card-red"
                st.markdown(f"""
                <div class="metric-card {prob_color}">
                    <h3>✅ Approval Probability</h3>
                    <div class="metric-value">{approval_prob:.1f}%</div>
                    <div class="metric-label">Enhanced AI Prediction</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Confidence score
                confidence = prediction_result.get('confidence', 0) * 100
                conf_color = "metric-card-green" if confidence >= 80 else "metric-card-orange" if confidence >= 60 else "metric-card-red"
                st.markdown(f"""
                <div class="metric-card {conf_color}">
                    <h3>🎯 Model Confidence</h3>
                    <div class="metric-value">{confidence:.1f}%</div>
                    <div class="metric-label">Prediction Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                # Final recommendation
                predicted_outcome = prediction_result.get('prediction', prediction_result.get('predicted_outcome', 'Unknown'))
                rec_color = "metric-card-green" if predicted_outcome == "Approved" else "metric-card-red"
                st.markdown(f"""
                <div class="metric-card {rec_color}">
                    <h3>💡 Final Decision</h3>
                    <div class="metric-value">{predicted_outcome}</div>
                    <div class="metric-label">Enhanced AI Decision</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show AI-driven analysis based purely on model outputs
            st.subheader("📊 AI Risk Assessment")
            
            # Display model-driven feature importance (no manual calculations)
            if 'feature_importance' in prediction_result:
                st.markdown("**🎯 AI Model Feature Impact Analysis:**")
                st.write("The following features had the highest influence on this prediction:")
                
                feature_importance = prediction_result['feature_importance']
                
                # Create a DataFrame for better visualization
                importance_df = pd.DataFrame([
                    {'Feature': feature.replace('_', ' ').title(), 
                     'Impact': importance * 100,
                     'Impact_Level': '🔴 Critical' if importance > 0.15 else '🟡 Moderate' if importance > 0.08 else '🟢 Minor'}
                    for feature, importance in feature_importance.items()
                ])
                
                # Display as a clean table
                for _, row in importance_df.iterrows():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{row['Feature']}**")
                    with col2:
                        st.write(f"{row['Impact']:.1f}%")
                    with col3:
                        st.write(row['Impact_Level'])
                
                # Show a plotly chart for feature importance
                fig = px.bar(
                    importance_df.head(6), 
                    x='Impact', 
                    y='Feature',
                    orientation='h',
                    title="Top AI Model Features (by Impact)",
                    color='Impact',
                    color_continuous_scale='RdYlGn_r'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Fallback to basic model confidence analysis
                st.markdown("**AI Model Analysis:**")
                model_confidence = prediction_result.get('confidence', 0) * 100
                approval_prob = prediction_result.get('approval_probability', 0) * 100
                
                if model_confidence >= 80:
                    st.success(f"🎯 **High Confidence Prediction** ({model_confidence:.1f}%)")
                    st.write("The AI model is very confident in this assessment based on the provided data.")
                elif model_confidence >= 60:
                    st.warning(f"⚠️ **Moderate Confidence Prediction** ({model_confidence:.1f}%)")
                    st.write("The AI model has moderate confidence. Additional verification may be beneficial.")
                else:
                    st.error(f"❓ **Low Confidence Prediction** ({model_confidence:.1f}%)")
                    st.write("The AI model has low confidence. Manual review strongly recommended.")
                
                # Show probability insights
                if approval_prob >= 70:
                    st.info("💡 **Strong approval indicators** detected by the model")
                elif approval_prob >= 40:
                    st.warning("🤔 **Mixed risk signals** - borderline case requiring careful review")
                else:
                    st.error("⚠️ **High risk factors** identified by the model")
            
            # Show denial reasons if denied or risk warnings if approved
            if predicted_outcome == "Denied":
                # Enhanced denial reasons from the new predictor
                denial_reasons = prediction_result.get('denial_reasons', [])
                primary_denial = prediction_result.get('primary_denial_reason', '')
                
                if denial_reasons:
                    st.error("**🚫 Enhanced AI Denial Analysis:**")
                    if primary_denial:
                        st.error(f"**Primary Reason:** {primary_denial}")
                    
                    st.error("**All Contributing Factors:**")
                    for i, reason in enumerate(denial_reasons, 1):
                        st.error(f"   {i}. {reason}")
                        
                    # Show denial explanations if available
                    denial_explanations = prediction_result.get('denial_explanations', {})
                    if denial_explanations:
                        with st.expander("🔍 Detailed Denial Analysis"):
                            for reason, explanation in denial_explanations.items():
                                st.write(f"**{reason}:** {explanation}")
                else:
                    # Fallback to old format
                    denial_reason = prediction_result.get('denial_reason', '')
                    if denial_reason:
                        st.error(f"**Primary Denial Reason:** {denial_reason}")
            else:
                # Show risk warnings for approved loans with high-risk factors
                risk_warnings = prediction_result.get('risk_warnings', [])
                if risk_warnings:
                    st.warning("**⚠️ Risk Factors to Monitor:**")
                    for i, warning in enumerate(risk_warnings, 1):
                        st.warning(f"   {i}. {warning}")
                elif approval_prob < 60:
                    st.warning("**⚠️ Borderline Approval:** Consider additional verification")
                elif approval_prob < 80:
                    st.info("**ℹ️ Standard Approval:** Normal processing recommended")
            
            # Show model insights - enhanced AI-driven analysis
            with st.expander("🔍 Enhanced AI Model Insights & Training Data"):
                st.markdown(f"""
                **Enhanced AI Model Information:**
                - **Algorithm:** Multi-Model Ensemble (Random Forest, XGBoost, LightGBM) with SMOTE balancing
                - **Training Source:** Real Kansas HMDA lending data (94,530+ records)
                - **Features:** Advanced feature engineering with {prediction_result.get('feature_count', 'N/A')} total features
                - **Model Focus:** DTI range parsing, financial ratios, and fair lending compliance
                - **Best Model:** {prediction_result.get('model_used', 'Unknown')}
                - **Class Balance:** SMOTE-enhanced training for realistic denial rates
                """)
                
                # Show model performance metrics if available
                model_metadata = prediction_result.get('model_metadata', {})
                if model_metadata:
                    st.markdown("**🎯 Model Performance:**")
                    st.write(f"• **Training Denial Rate:** {model_metadata.get('denial_rate', 'N/A')}")
                    st.write(f"• **Training Samples:** {model_metadata.get('training_samples', 'N/A'):,}")
                    st.write(f"• **Model Confidence:** {confidence:.1f}%")
                
                # Show actual inputs processed by enhanced AI model
                st.markdown("**🔢 Enhanced Feature Processing:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"• **DTI Ratio:** {application_data.get('debt_to_income_ratio', 'N/A')}%")
                    st.write(f"• **LTV Ratio:** {application_data.get('loan_to_value_ratio', 'N/A')}%")
                    st.write(f"• **Credit Score:** {application_data.get('applicant_credit_score_type', 'N/A')}")
                with col2:
                    st.write(f"• **Property Value:** ${application_data.get('property_value', 'N/A'):,}")
                    st.write(f"• **Loan Amount:** ${application_data.get('loan_amount', 'N/A'):,}")
                    st.write(f"• **Annual Income:** ${application_data.get('income', 'N/A'):,}")
                
                st.markdown(f"""
                **Enhanced AI Prediction Methodology:**
                This system uses state-of-the-art machine learning trained on real lending data:
                1. **Advanced Feature Engineering** - DTI range parsing, financial ratios, risk scoring
                2. **Class Imbalance Handling** - SMOTE balancing for realistic predictions
                3. **Multi-Model Ensemble** - Best performing models selected via cross-validation
                4. **Fair Lending Compliance** - Demographic bias monitoring and HMDA compliance
                5. **HMDA-Compliant Denial Reasons** - Official reason codes with intelligent prioritization
                
                **Model Improvements Over Legacy System:**
                - ✅ Fixed "always approved" predictions through SMOTE balancing
                - ✅ Enhanced DTI parsing for complex HMDA formats ("20%-<30%")
                - ✅ Advanced feature engineering (financial ratios, risk scores)
                - ✅ Multiple model comparison with hyperparameter tuning
                - ✅ Fair lending compliance monitoring
                """)
                
                st.markdown(f"""
                **Model Transparency & Compliance:**
                This enhanced AI system is designed for production lending with:
                - ✅ Advanced fair lending practices and ECOA compliance
                - ✅ HMDA reporting requirement adherence with improved accuracy
                - ✅ Explainable AI for regulatory transparency
                - ✅ Real-world validation using actual Kansas lending data
                - ✅ Continuous model monitoring and bias detection
                - ✅ Financial risk prioritization over demographic factors
                """)
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please ensure all fields are filled correctly and try again.")


def render_filters_sidebar(df):
    """Render comprehensive sidebar filters"""
    st.sidebar.header("🎛️ Analysis Filters")
    
    # Opportunity level filter
    opportunity_levels = ['All'] + sorted(df['opportunity_level'].unique().tolist())
    selected_level = st.sidebar.selectbox(
        "Opportunity Level:",
        opportunity_levels
    )
    
    # Score range filter
    min_score, max_score = st.sidebar.slider(
        "Opportunity Score Range:",
        float(df['opportunity_score'].min()),
        float(df['opportunity_score'].max()),
        (float(df['opportunity_score'].min()), 
         float(df['opportunity_score'].max()))
    )
    
    # Population filter
    if 'total_population' in df.columns:
        min_pop = int(df['total_population'].min())
        max_pop = int(df['total_population'].max())
        pop_range = st.sidebar.slider(
            "Population Range:",
            min_value=min_pop,
            max_value=max_pop,
            value=(min_pop, max_pop)
        )
    else:
        pop_range = (0, 100000)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_level != 'All':
        filtered_df = filtered_df[filtered_df['opportunity_level'] == selected_level]
    
    filtered_df = filtered_df[
        (filtered_df['opportunity_score'] >= min_score) &
        (filtered_df['opportunity_score'] <= max_score)
    ]
    
    if 'total_population' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['total_population'] >= pop_range[0]) &
            (filtered_df['total_population'] <= pop_range[1])
        ]
    
    # Display filter summary
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Filter Summary")
    st.sidebar.write(f"**Showing:** {len(filtered_df)} of {len(df)} tracts")
    if len(filtered_df) > 0:
        st.sidebar.write(f"**Avg Score:** {filtered_df['opportunity_score'].mean():.1f}")
        st.sidebar.write(f"**Total Pop:** {filtered_df['total_population'].sum():,}")
    
    return filtered_df


def create_temporal_forecasting_view(config):
    """Create temporal forecasting analysis view using pre-computed results"""
    
    st.title("🔮 Temporal Opportunity Forecasting")
    st.markdown("""
    **Multi-Year Analysis & Future Predictions**
    
    This analysis uses pre-computed temporal forecasting models trained on historical HMDA data (2022-2024) 
    to provide opportunity score predictions for 2025 and 2026.
    """)
    
    # Load pre-computed temporal forecasts
    with st.spinner("� Loading pre-computed temporal forecasting results..."):
        forecaster, results = load_temporal_forecasts(config)
    
    if forecaster is None or results is None:
        st.warning("⚠️ Temporal forecasting results not available.")
        st.info("Run the temporal forecasting pipeline to generate results:")
        st.code("python3 src/temporal_forecasting_pipeline.py")
        return
    
    if not results.get('success', False):
        st.error(f"Temporal forecasting failed: {results.get('error', 'Unknown error')}")
        return
    
    # Display data freshness info
    data_timestamp = results.get('data_timestamp', 'Unknown')
    if data_timestamp != 'Unknown':
        st.info(f"📅 **Data Generated:** {data_timestamp[:19]} | **Status:** Pre-computed results loaded instantly")
    
    # Display summary metrics
    st.subheader("📊 Forecasting Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Years of Historical Data",
            value=f"{results.get('yearly_data_loaded', 0)} years",
            help="Number of years with historical lending data"
        )
    
    with col2:
        st.metric(
            label="Models Trained",
            value=f"{results.get('models_trained', 0)}",
            help="Number of year-specific opportunity scoring models"
        )
    
    with col3:
        st.metric(
            label="Future Predictions",
            value=f"{results.get('predictions_generated', 0)} years",
            help="Number of future years predicted (2025, 2026)"
        )
    
    with col4:
        st.metric(
            label="Timeline Records",
            value=f"{results.get('timeline_records', 0):,}",
            help="Total records in comprehensive timeline"
        )
    
    # Model Performance Section
    st.subheader("🎯 Model Performance by Year")
    
    if 'summary' in results and 'models_performance' in results['summary']:
        performance_data = []
        
        for year, perf in results['summary']['models_performance'].items():
            performance_data.append({
                'Year': year,
                'Best Model': perf['best_model'].replace('_', ' ').title(),
                'R² Score': f"{perf['r2_score']:.3f}",
                'Mean Absolute Error': f"{perf['mae']:.3f}"
            })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df, use_container_width=True)
        
        # Performance visualization
        if len(performance_data) > 0:
            fig_perf = px.bar(
                perf_df, 
                x='Year', 
                y='R² Score',
                title="Model Performance by Training Year",
                color='Best Model',
                text='R² Score'
            )
            fig_perf.update_traces(texttemplate='%{text}', textposition='outside')
            fig_perf.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig_perf, use_container_width=True)
    
    # Future Predictions Section
    st.subheader("🔮 Future Opportunity Predictions")
    
    if forecaster and 'predictions' in forecaster and forecaster['predictions']:
        
        # Create tabs for different prediction years
        pred_years = sorted(list(forecaster['predictions'].keys()))
        if len(pred_years) > 1:
            tabs = st.tabs([f"📅 {year}" for year in pred_years])
            
            for i, year in enumerate(pred_years):
                with tabs[i]:
                    display_prediction_year(forecaster['predictions'][year], year)
        else:
            # Single year display
            year = pred_years[0]
            display_prediction_year(forecaster['predictions'][year], year)
    
    # Historical vs Predicted Timeline
    st.subheader("📈 Historical vs Predicted Timeline")
    
    if forecaster and 'yearly_scores' in forecaster and 'predictions' in forecaster:
        create_timeline_visualization(forecaster)
    
    # Top Opportunities by Prediction
    st.subheader("🏆 Top Future Opportunities")
    
    if forecaster and 'predictions' in forecaster:
        create_future_opportunities_analysis(forecaster['predictions'])
    
    # Performance and refresh info
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚡ Performance Benefits")
        st.markdown("""
        **Pre-computed Results:**
        - ✅ **Instant Loading** - No real-time computation
        - ✅ **Consistent Results** - Same predictions across sessions  
        - ✅ **Reliable Performance** - No browser timeouts
        - ✅ **Resource Efficient** - No CPU-intensive calculations
        """)
    
    with col2:
        st.markdown("### 🔄 Data Refresh")
        st.markdown(f"""
        **Current Data Status:**
        - **Generated:** {data_timestamp[:19] if data_timestamp != 'Unknown' else 'Unknown'}
        - **Source:** Pre-computed pipeline results
        - **Models:** XGBoost ensemble (98.9%+ accuracy)
        
        **To refresh data:** Run temporal forecasting pipeline
        """)
        
        if st.button("🔄 Refresh Forecasting Data"):
            with st.spinner("Running temporal forecasting pipeline..."):
                try:
                    from temporal_forecasting_pipeline import run_temporal_forecasting_pipeline
                    pipeline_results = run_temporal_forecasting_pipeline()
                    
                    if pipeline_results['success']:
                        st.success("✅ Forecasting data refreshed successfully!")
                        st.info("Please refresh the page to see updated results.")
                    else:
                        st.error(f"❌ Pipeline failed: {pipeline_results['error']}")
                except Exception as e:
                    st.error(f"❌ Error running pipeline: {str(e)}")
                    st.info("💡 Alternatively, run in terminal: `python3 src/temporal_forecasting_pipeline.py`")


def display_prediction_year(pred_data, year):
    """Display predictions for a specific year"""
    
    if pred_data.empty:
        st.warning(f"No prediction data available for {year}")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_score = pred_data['predicted_opportunity_score'].mean()
        st.metric(
            label="Average Predicted Score",
            value=f"{avg_score:.1f}",
            help=f"Average opportunity score predicted for {year}"
        )
    
    with col2:
        high_opportunity = len(pred_data[pred_data['predicted_opportunity_score'] >= 70])
        st.metric(
            label="High Opportunity Tracts",
            value=f"{high_opportunity}",
            help="Census tracts with predicted scores ≥ 70"
        )
    
    with col3:
        avg_confidence = pred_data['prediction_confidence'].mean()
        st.metric(
            label="Average Confidence",
            value=f"{avg_confidence:.1f}%",
            help="Average prediction confidence level"
        )
    
    # Distribution visualization
    fig_dist = px.histogram(
        pred_data,
        x='predicted_opportunity_score',
        nbins=20,
        title=f"Distribution of Predicted Opportunity Scores ({year})",
        labels={'predicted_opportunity_score': 'Predicted Opportunity Score', 'count': 'Number of Census Tracts'}
    )
    fig_dist.update_layout(height=400)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Top predictions table
    st.write("**Top 10 Predicted Opportunities:**")
    top_predictions = pred_data.nlargest(10, 'predicted_opportunity_score')[
        ['census_tract', 'predicted_opportunity_score', 'prediction_confidence', 'trend_direction']
    ].round(2)
    st.dataframe(top_predictions, use_container_width=True)


def create_timeline_visualization(forecaster):
    """Create comprehensive timeline visualization"""
    
    timeline_data = []
    
    # Add historical data
    if 'yearly_scores' in forecaster:
        for year, data in forecaster['yearly_scores'].items():
            # Sample some census tracts for visualization
            sample_tracts = data['census_tract'].unique()[:10]  # Show top 10 for clarity
            sample_data = data[data['census_tract'].isin(sample_tracts)]
            
            for _, row in sample_data.iterrows():
                timeline_data.append({
                    'census_tract': row['census_tract'],
                    'year': year,
                    'opportunity_score': row['opportunity_score'],
                    'data_type': 'Historical'
                })
    
    # Add prediction data
    if 'predictions' in forecaster:
        for year, pred_data in forecaster['predictions'].items():
            # Same sample tracts
            if timeline_data:  # If we have historical data
                sample_tracts = list(set([item['census_tract'] for item in timeline_data]))
                sample_pred = pred_data[pred_data['census_tract'].isin(sample_tracts)]
            else:
                sample_pred = pred_data.head(10)
            
            for _, row in sample_pred.iterrows():
                timeline_data.append({
                    'census_tract': row['census_tract'],
                    'year': year,
                    'opportunity_score': row['predicted_opportunity_score'],
                    'data_type': 'Predicted'
                })
    
    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data)
        
        # Create line chart
        fig_timeline = px.line(
            timeline_df,
            x='year',
            y='opportunity_score',
            color='census_tract',
            line_dash='data_type',
            title="Opportunity Score Timeline: Historical vs Predicted",
            labels={'opportunity_score': 'Opportunity Score', 'year': 'Year'}
        )
        
        # Add vertical line to separate historical from predicted
        fig_timeline.add_vline(
            x=2024.5, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Prediction Start"
        )
        
        fig_timeline.update_layout(height=500)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Summary statistics
        st.write("**Timeline Statistics:**")
        summary_stats = timeline_df.groupby(['year', 'data_type'])['opportunity_score'].agg(['mean', 'std', 'count']).round(2)
        st.dataframe(summary_stats, use_container_width=True)


def create_future_opportunities_analysis(predictions):
    """Analyze and display future opportunities"""
    
    all_predictions = []
    
    for year, pred_data in predictions.items():
        pred_copy = pred_data.copy()
        pred_copy['prediction_year'] = year
        all_predictions.append(pred_copy)
    
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        
        # Find consistently high-performing tracts
        st.write("**Consistently High Opportunity Census Tracts:**")
        
        high_threshold = 70
        tract_performance = combined_predictions.groupby('census_tract').agg({
            'predicted_opportunity_score': ['mean', 'min', 'max'],
            'prediction_confidence': 'mean'
        }).round(2)
        
        # Flatten column names
        tract_performance.columns = ['Avg_Score', 'Min_Score', 'Max_Score', 'Avg_Confidence']
        
        # Filter for consistently high performers
        high_performers = tract_performance[
            (tract_performance['Avg_Score'] >= high_threshold) &
            (tract_performance['Min_Score'] >= high_threshold * 0.8)
        ].sort_values('Avg_Score', ascending=False)
        
        if not high_performers.empty:
            st.dataframe(high_performers.head(15), use_container_width=True)
            
            # Visualization of top performers
            top_tracts = high_performers.head(5).index.tolist()
            top_data = combined_predictions[combined_predictions['census_tract'].isin(top_tracts)]
            
            fig_top = px.bar(
                top_data,
                x='census_tract',
                y='predicted_opportunity_score',
                color='prediction_year',
                title="Top 5 Consistently High-Opportunity Census Tracts",
                barmode='group'
            )
            fig_top.update_layout(height=400)
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("No census tracts meet the high-opportunity criteria across all prediction years.")


def main():
    """Main dashboard application"""
    
    # Load data
    with st.spinner("Loading lending opportunity data..."):
        scored_data, config = load_all_data()
    
    if scored_data is None or config is None:
        st.error("Failed to load data. Please check the data files and try again.")
        return
    
    # Sidebar navigation
    st.sidebar.title("🏦 Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis View:",
        ["Overview", "Component Analysis", "Market Segmentation", "Year-over-Year Analysis", "Temporal Forecasting", "Census Tract Details", "Loan Prediction Demo"]
    )
    
    # Apply filters
    filtered_data = render_filters_sidebar(scored_data)
    
    # Display selected page
    if page == "Overview":
        create_opportunity_overview(filtered_data)
        
    elif page == "Component Analysis":
        create_component_analysis(filtered_data)
        
    elif page == "Market Segmentation":
        with st.spinner("Loading market segmentation..."):
            segmented_data, analysis_results = load_market_segments_uncached(filtered_data, config)
        create_market_segmentation_view(segmented_data, analysis_results)
        
    elif page == "Year-over-Year Analysis":
        if ENHANCED_YOY_AVAILABLE:
            create_enhanced_yoy_dashboard()
        else:
            st.error("Enhanced Year-over-Year analysis not available. Please check imports.")
            st.info("Ensure enhanced_yoy_dashboard.py and enhanced_yoy_analyzer.py are properly installed.")
        
    elif page == "Temporal Forecasting":
        create_temporal_forecasting_view(config)
        
    elif page == "Census Tract Details":
        create_tract_details_view(filtered_data, config)
        
    elif page == "Loan Prediction Demo":
        create_loan_prediction_demo()
    
    # Footer with data summary
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Data Summary")
    st.sidebar.write(f"Census Tracts: {len(filtered_data)}")
    st.sidebar.write(f"Avg Opportunity Score: {filtered_data['opportunity_score'].mean():.1f}")
    st.sidebar.write(f"Total Population: {filtered_data['total_population'].sum():,}")
    
    # Download data
    if st.sidebar.button("📥 Download Data"):
        csv = filtered_data.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"lending_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()