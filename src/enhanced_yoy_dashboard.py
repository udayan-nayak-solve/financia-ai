#!/usr/bin/env python3
"""
Enhanced Year-over-Year Dashboard Component

Provides comprehensive year-over-year census tract analysis with
business-focused insights and visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime
from enhanced_yoy_analyzer import YearOverYearAnalyzer


def create_enhanced_yoy_dashboard():
    """Create comprehensive year-over-year analysis dashboard"""
    
    st.markdown('<div class="main-header">📈 Year-over-Year Census Tract Performance Analysis</div>', 
                unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = YearOverYearAnalyzer()
    
    # Load and analyze data
    with st.spinner("Loading multi-year HMDA data and performing analysis..."):
        try:
            analyzer.load_multi_year_data()
            yoy_analysis = analyzer.perform_yoy_analysis()
            business_insights = analyzer.generate_business_insights()
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return
    
    if not yoy_analysis:
        st.error("No year-over-year data available for analysis")
        return
    
    # Executive Summary Section
    create_executive_summary_section(business_insights['executive_summary'])
    
    # Create main tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Overview", 
        "🏆 Performance Rankings", 
        "📍 Census Tract Deep Dive",
        "🎯 Market Segments",
        "💡 Strategic Insights"
    ])
    
    with tab1:
        create_market_overview_tab(yoy_analysis, business_insights)
    
    with tab2:
        create_performance_rankings_tab(yoy_analysis)
    
    with tab3:
        create_tract_deep_dive_tab(yoy_analysis)
    
    with tab4:
        create_market_segments_tab(yoy_analysis)
    
    with tab5:
        create_strategic_insights_tab(business_insights)


def create_executive_summary_section(exec_summary: dict):
    """Create executive summary section with key metrics"""
    
    st.subheader("🎯 Executive Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Analysis Period",
            value=exec_summary['analysis_period'],
            help="Years included in the analysis"
        )
    
    with col2:
        st.metric(
            label="Census Tracts",
            value=f"{exec_summary['total_tracts_analyzed']:,}",
            help="Total number of census tracts analyzed"
        )
    
    with col3:
        volume_trend = exec_summary['volume_trend']
        trend_color = "normal" if abs(volume_trend) < 5 else ("inverse" if volume_trend < 0 else "normal")
        st.metric(
            label="Volume Trend",
            value=f"{volume_trend:+.1f}%",
            delta=f"Year-over-year change",
            delta_color=trend_color,
            help="Overall application volume trend"
        )
    
    with col4:
        approval_trend = exec_summary['approval_rate_trend']
        approval_color = "normal" if approval_trend > 0 else "inverse"
        st.metric(
            label="Approval Rate Trend",
            value=f"{approval_trend:+.1f}%",
            delta="Percentage points",
            delta_color=approval_color,
            help="Overall approval rate trend"
        )
    
    with col5:
        st.metric(
            label="Dominant Segment",
            value=exec_summary['dominant_market_segment'],
            help="Market segment with highest volume"
        )


def create_market_overview_tab(yoy_analysis: dict, business_insights: dict):
    """Create market overview tab with trend visualizations"""
    
    st.subheader("📊 Market Performance Overview")
    
    # Overall market trends chart
    trend_data = yoy_analysis['trend_analysis']['yearly_performance']
    
    if not trend_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Volume trends
            fig_volume = px.line(
                x=trend_data.index,
                y=trend_data['total_applications'],
                title="📈 Application Volume Trends",
                labels={'x': 'Year', 'y': 'Total Applications'},
                markers=True
            )
            fig_volume.update_traces(line_color='#2E86AB', line_width=3)
            fig_volume.update_layout(height=400)
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with col2:
            # Approval rate trends
            fig_approval = px.line(
                x=trend_data.index,
                y=trend_data['approval_rate'],
                title="✅ Approval Rate Trends",
                labels={'x': 'Year', 'y': 'Approval Rate (%)'},
                markers=True
            )
            fig_approval.update_traces(line_color='#A23B72', line_width=3)
            fig_approval.update_layout(height=400)
            st.plotly_chart(fig_approval, use_container_width=True)
    
    # Financial metrics trends
    st.subheader("💰 Financial Metrics Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not trend_data.empty and 'avg_loan_amount' in trend_data.columns:
            fig_loan = px.bar(
                x=trend_data.index,
                y=trend_data['avg_loan_amount'],
                title="💵 Average Loan Amount by Year",
                labels={'x': 'Year', 'y': 'Average Loan Amount ($)'},
                color=trend_data['avg_loan_amount'],
                color_continuous_scale='Viridis'
            )
            fig_loan.update_layout(height=400)
            st.plotly_chart(fig_loan, use_container_width=True)
    
    with col2:
        if not trend_data.empty and 'total_loan_volume' in trend_data.columns:
            fig_volume_dollars = px.bar(
                x=trend_data.index,
                y=trend_data['total_loan_volume'] / 1e9,  # Convert to billions
                title="📊 Total Loan Volume by Year",
                labels={'x': 'Year', 'y': 'Total Volume ($ Billions)'},
                color=trend_data['total_loan_volume'],
                color_continuous_scale='Blues'
            )
            fig_volume_dollars.update_layout(height=400)
            st.plotly_chart(fig_volume_dollars, use_container_width=True)
    
    # Growth and Emerging Markets
    st.subheader("🚀 Growth & Emerging Markets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏆 Top Growth Leaders**")
        growth_leaders = yoy_analysis['trend_analysis']['top_growth_tracts']
        
        if growth_leaders:
            growth_df = pd.DataFrame(growth_leaders)
            growth_df['Volume Growth'] = growth_df['volume_growth_pct'].apply(lambda x: f"{x:.1f}%")
            growth_df['Approval Rate'] = growth_df['current_approval_rate'].apply(lambda x: f"{x:.1f}%")
            
            display_df = growth_df[['census_tract', 'Volume Growth', 'Approval Rate', 'market_segment']].head(10)
            display_df.columns = ['Census Tract', 'Growth Rate', 'Approval Rate', 'Market Segment']
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No significant growth leaders identified")
    
    with col2:
        st.markdown("**🌱 Emerging Markets**")
        emerging_markets = yoy_analysis['trend_analysis']['emerging_markets']
        
        if emerging_markets:
            emerging_df = pd.DataFrame(emerging_markets)
            emerging_df['Volume Growth'] = emerging_df['volume_growth_pct'].apply(lambda x: f"{x:.1f}%")
            emerging_df['Approval Rate'] = emerging_df['current_approval_rate'].apply(lambda x: f"{x:.1f}%")
            emerging_df['Avg Loan'] = emerging_df['avg_loan_amount'].apply(lambda x: f"${x:,.0f}")
            
            display_df = emerging_df[['census_tract', 'current_volume', 'Approval Rate', 'market_segment']].head(10)
            display_df.columns = ['Census Tract', 'Current Volume', 'Approval Rate', 'Market Segment']
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No emerging markets identified")


def create_performance_rankings_tab(yoy_analysis: dict):
    """Create performance rankings tab"""
    
    st.subheader("🏆 Performance Rankings")
    
    rankings = yoy_analysis['performance_rankings']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Top Improvers")
        
        if rankings['top_improvers']:
            improvers_df = pd.DataFrame(rankings['top_improvers'])
            
            # Create improvement score visualization
            improvers_df['improvement_display'] = (
                improvers_df['volume_change_pct'] * 0.4 + 
                improvers_df['approval_rate_change'] * 0.6
            )
            
            fig_improvers = px.bar(
                improvers_df.head(10),
                x='improvement_display',
                y='census_tract',
                orientation='h',
                title="Improvement Score (Volume 40% + Approval Rate 60%)",
                labels={'improvement_display': 'Improvement Score', 'census_tract': 'Census Tract'},
                color='improvement_display',
                color_continuous_scale='Greens'
            )
            fig_improvers.update_layout(height=400)
            st.plotly_chart(fig_improvers, use_container_width=True)
            
            # Detailed table
            display_cols = ['census_tract', 'volume_change_pct', 'approval_rate_change', 
                          'current_volume', 'current_approval_rate', 'market_segment']
            display_df = improvers_df[display_cols].head(10).copy()
            
            display_df['Volume Change'] = display_df['volume_change_pct'].apply(lambda x: f"{x:+.1f}%")
            display_df['Approval Change'] = display_df['approval_rate_change'].apply(lambda x: f"{x:+.1f}%")
            display_df['Current Volume'] = display_df['current_volume']
            display_df['Current Approval'] = display_df['current_approval_rate'].apply(lambda x: f"{x:.1f}%")
            
            final_df = display_df[['census_tract', 'Volume Change', 'Approval Change', 
                                 'Current Volume', 'Current Approval', 'market_segment']]
            final_df.columns = ['Census Tract', 'Volume Δ', 'Approval Δ', 'Volume', 'Approval', 'Segment']
            
            st.dataframe(final_df, use_container_width=True)
        else:
            st.info("No improvement data available")
    
    with col2:
        st.markdown("### 📉 Areas of Concern")
        
        if rankings['top_decliners']:
            decliners_df = pd.DataFrame(rankings['top_decliners'])
            
            # Create decline score visualization
            decliners_df['decline_display'] = (
                decliners_df['volume_change_pct'] * 0.4 + 
                decliners_df['approval_rate_change'] * 0.6
            )
            
            fig_decliners = px.bar(
                decliners_df.head(10),
                x='decline_display',
                y='census_tract',
                orientation='h',
                title="Decline Score (Volume 40% + Approval Rate 60%)",
                labels={'decline_display': 'Decline Score', 'census_tract': 'Census Tract'},
                color='decline_display',
                color_continuous_scale='Reds'
            )
            fig_decliners.update_layout(height=400)
            st.plotly_chart(fig_decliners, use_container_width=True)
            
            # Detailed table
            display_cols = ['census_tract', 'volume_change_pct', 'approval_rate_change', 
                          'current_volume', 'current_approval_rate', 'market_segment']
            display_df = decliners_df[display_cols].head(10).copy()
            
            display_df['Volume Change'] = display_df['volume_change_pct'].apply(lambda x: f"{x:+.1f}%")
            display_df['Approval Change'] = display_df['approval_rate_change'].apply(lambda x: f"{x:+.1f}%")
            display_df['Current Volume'] = display_df['current_volume']
            display_df['Current Approval'] = display_df['current_approval_rate'].apply(lambda x: f"{x:.1f}%")
            
            final_df = display_df[['census_tract', 'Volume Change', 'Approval Change', 
                                 'Current Volume', 'Current Approval', 'market_segment']]
            final_df.columns = ['Census Tract', 'Volume Δ', 'Approval Δ', 'Volume', 'Approval', 'Segment']
            
            st.dataframe(final_df, use_container_width=True)
        else:
            st.info("No decline data available")


def create_tract_deep_dive_tab(yoy_analysis: dict):
    """Create census tract deep dive analysis"""
    
    st.subheader("📍 Census Tract Deep Dive")
    
    # Get all available census tracts
    yearly_metrics = yoy_analysis['yearly_metrics']
    
    if not yearly_metrics:
        st.error("No yearly metrics data available")
        return
    
    # Combine all census tracts from all years
    all_tracts = set()
    for year_df in yearly_metrics.values():
        all_tracts.update(year_df['census_tract'].unique())
    
    all_tracts = sorted(list(all_tracts))
    
    # Tract selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_tract = st.selectbox(
            "Select Census Tract for Analysis:",
            all_tracts,
            help="Choose a census tract to see detailed year-over-year performance"
        )
    
    with col2:
        # Filter options
        metric_focus = st.selectbox(
            "Focus Metric:",
            ["All Metrics", "Volume", "Approval Rate", "Financial", "Market Position"]
        )
    
    if selected_tract:
        create_individual_tract_analysis(selected_tract, yoy_analysis, metric_focus)


def create_individual_tract_analysis(tract: str, yoy_analysis: dict, metric_focus: str):
    """Create detailed analysis for individual census tract"""
    
    # Collect data for this tract across all years
    tract_data = []
    yearly_metrics = yoy_analysis['yearly_metrics']
    
    for year, year_df in yearly_metrics.items():
        tract_year_data = year_df[year_df['census_tract'] == tract]
        if not tract_year_data.empty:
            data_dict = tract_year_data.iloc[0].to_dict()
            tract_data.append(data_dict)
    
    if not tract_data:
        st.error(f"No data found for census tract {tract}")
        return
    
    tract_df = pd.DataFrame(tract_data).sort_values('year')
    
    # Summary metrics
    st.markdown(f"### 📊 Census Tract {tract} - Performance Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        years_available = len(tract_df)
        st.metric("Years of Data", years_available)
    
    with col2:
        latest_volume = tract_df['total_applications'].iloc[-1]
        if len(tract_df) > 1:
            previous_volume = tract_df['total_applications'].iloc[-2]
            volume_change = ((latest_volume - previous_volume) / previous_volume * 100) if previous_volume > 0 else 0
            st.metric("Current Volume", f"{latest_volume:,}", delta=f"{volume_change:+.1f}%")
        else:
            st.metric("Current Volume", f"{latest_volume:,}")
    
    with col3:
        latest_approval = tract_df['approval_rate'].iloc[-1]
        if len(tract_df) > 1:
            previous_approval = tract_df['approval_rate'].iloc[-2]
            approval_change = latest_approval - previous_approval
            st.metric("Current Approval Rate", f"{latest_approval:.1f}%", delta=f"{approval_change:+.1f}%")
        else:
            st.metric("Current Approval Rate", f"{latest_approval:.1f}%")
    
    with col4:
        latest_segment = tract_df['market_segment'].iloc[-1]
        st.metric("Market Segment", latest_segment)
    
    with col5:
        latest_tier = tract_df['performance_tier'].iloc[-1]
        st.metric("Performance Tier", latest_tier)
    
    # Create visualizations based on focus
    if metric_focus == "All Metrics" or metric_focus == "Volume":
        create_volume_analysis_charts(tract_df, tract)
    
    if metric_focus == "All Metrics" or metric_focus == "Approval Rate":
        create_approval_rate_analysis_charts(tract_df, tract)
    
    if metric_focus == "All Metrics" or metric_focus == "Financial":
        create_financial_analysis_charts(tract_df, tract)
    
    # Detailed data table
    st.markdown("### 📋 Year-over-Year Detailed Data")
    
    display_df = tract_df.copy()
    
    # Format columns for display
    display_df['Year'] = display_df['year'].astype(int)
    display_df['Applications'] = display_df['total_applications']
    display_df['Originations'] = display_df['total_originations']
    display_df['Approval Rate'] = display_df['approval_rate'].apply(lambda x: f"{x:.1f}%")
    display_df['Avg Loan Amount'] = display_df['avg_loan_amount'].apply(lambda x: f"${x:,.0f}")
    display_df['Total Volume'] = display_df['total_loan_volume'].apply(lambda x: f"${x:,.0f}")
    display_df['Market Segment'] = display_df['market_segment']
    display_df['Performance Tier'] = display_df['performance_tier']
    
    cols_to_show = ['Year', 'Applications', 'Originations', 'Approval Rate', 
                   'Avg Loan Amount', 'Total Volume', 'Market Segment', 'Performance Tier']
    
    st.dataframe(display_df[cols_to_show], use_container_width=True)


def create_volume_analysis_charts(tract_df: pd.DataFrame, tract: str):
    """Create volume analysis charts"""
    
    st.markdown("#### 📊 Volume Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Application volume over time
        fig_volume = px.line(
            tract_df,
            x='year',
            y='total_applications',
            title=f"Application Volume - Tract {tract}",
            markers=True,
            labels={'year': 'Year', 'total_applications': 'Total Applications'}
        )
        fig_volume.update_traces(line_color='#2E86AB', line_width=3, marker_size=8)
        fig_volume.update_layout(height=400)
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with col2:
        # Originations vs applications
        fig_orig = go.Figure()
        
        fig_orig.add_trace(go.Scatter(
            x=tract_df['year'],
            y=tract_df['total_applications'],
            mode='lines+markers',
            name='Applications',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8)
        ))
        
        fig_orig.add_trace(go.Scatter(
            x=tract_df['year'],
            y=tract_df['total_originations'],
            mode='lines+markers',
            name='Originations',
            line=dict(color='#A23B72', width=3),
            marker=dict(size=8)
        ))
        
        fig_orig.update_layout(
            title=f"Applications vs Originations - Tract {tract}",
            xaxis_title='Year',
            yaxis_title='Count',
            height=400
        )
        
        st.plotly_chart(fig_orig, use_container_width=True)


def create_approval_rate_analysis_charts(tract_df: pd.DataFrame, tract: str):
    """Create approval rate analysis charts"""
    
    st.markdown("#### ✅ Approval Rate Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Approval rate over time
        fig_approval = px.line(
            tract_df,
            x='year',
            y='approval_rate',
            title=f"Approval Rate Trend - Tract {tract}",
            markers=True,
            labels={'year': 'Year', 'approval_rate': 'Approval Rate (%)'}
        )
        fig_approval.update_traces(line_color='#28a745', line_width=3, marker_size=8)
        fig_approval.update_layout(height=400)
        st.plotly_chart(fig_approval, use_container_width=True)
    
    with col2:
        # Approval rates with performance tiers
        fig_tier = px.scatter(
            tract_df,
            x='year',
            y='approval_rate',
            color='performance_tier',
            size='total_applications',
            title=f"Approval Rate by Performance Tier - Tract {tract}",
            labels={'year': 'Year', 'approval_rate': 'Approval Rate (%)'}
        )
        fig_tier.update_layout(height=400)
        st.plotly_chart(fig_tier, use_container_width=True)


def create_financial_analysis_charts(tract_df: pd.DataFrame, tract: str):
    """Create financial metrics analysis charts"""
    
    st.markdown("#### 💰 Financial Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average loan amount over time
        fig_loan = px.line(
            tract_df,
            x='year',
            y='avg_loan_amount',
            title=f"Average Loan Amount - Tract {tract}",
            markers=True,
            labels={'year': 'Year', 'avg_loan_amount': 'Average Loan Amount ($)'}
        )
        fig_loan.update_traces(line_color='#fd7e14', line_width=3, marker_size=8)
        fig_loan.update_layout(height=400)
        st.plotly_chart(fig_loan, use_container_width=True)
    
    with col2:
        # Total loan volume over time
        fig_total_volume = px.bar(
            tract_df,
            x='year',
            y='total_loan_volume',
            title=f"Total Loan Volume - Tract {tract}",
            labels={'year': 'Year', 'total_loan_volume': 'Total Volume ($)'},
            color='market_segment'
        )
        fig_total_volume.update_layout(height=400)
        st.plotly_chart(fig_total_volume, use_container_width=True)


def create_market_segments_tab(yoy_analysis: dict):
    """Create market segments analysis tab"""
    
    st.subheader("🎯 Market Segment Analysis")
    
    segment_analysis = yoy_analysis['segment_analysis']
    
    if not segment_analysis:
        st.error("No market segment analysis data available")
        return
    
    # Segment performance overview
    st.markdown("### 📊 Segment Performance Overview")
    
    segment_summary = []
    for segment, data in segment_analysis.items():
        segment_summary.append({
            'Market Segment': segment,
            'Volume Trend': f"{data['volume_trend_pct']:.1f}%",
            'Approval Trend': f"{data['approval_rate_trend']:.1f}%",
            'Tract Count': data['tract_count'],
            'Avg Performance': data['avg_performance_tier']
        })
    
    segment_df = pd.DataFrame(segment_summary)
    st.dataframe(segment_df, use_container_width=True)
    
    # Detailed segment analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Volume trends by segment
        volume_data = []
        for segment, data in segment_analysis.items():
            yearly_perf = data['yearly_performance']
            for year, row in yearly_perf.iterrows():
                volume_data.append({
                    'Segment': segment,
                    'Year': year,
                    'Volume': row['total_applications']
                })
        
        if volume_data:
            volume_df = pd.DataFrame(volume_data)
            fig_segment_volume = px.line(
                volume_df,
                x='Year',
                y='Volume',
                color='Segment',
                title="Application Volume by Market Segment",
                markers=True
            )
            fig_segment_volume.update_layout(height=400)
            st.plotly_chart(fig_segment_volume, use_container_width=True)
    
    with col2:
        # Approval rates by segment
        approval_data = []
        for segment, data in segment_analysis.items():
            yearly_perf = data['yearly_performance']
            for year, row in yearly_perf.iterrows():
                approval_data.append({
                    'Segment': segment,
                    'Year': year,
                    'Approval Rate': row['approval_rate']
                })
        
        if approval_data:
            approval_df = pd.DataFrame(approval_data)
            fig_segment_approval = px.line(
                approval_df,
                x='Year',
                y='Approval Rate',
                color='Segment',
                title="Approval Rate by Market Segment",
                markers=True
            )
            fig_segment_approval.update_layout(height=400)
            st.plotly_chart(fig_segment_approval, use_container_width=True)


def create_strategic_insights_tab(business_insights: dict):
    """Create strategic insights and recommendations tab"""
    
    st.subheader("💡 Strategic Insights & Recommendations")
    
    # Growth opportunities
    st.markdown("### 🚀 Growth Opportunities")
    
    opportunities = business_insights['growth_opportunities']
    
    if opportunities:
        opp_df = pd.DataFrame(opportunities)
        
        # Categorize by opportunity type
        high_growth = opp_df[opp_df['type'] == 'High Growth Leader']
        emerging = opp_df[opp_df['type'] == 'Emerging Market']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not high_growth.empty:
                st.markdown("**🏆 High Growth Leaders**")
                display_cols = ['census_tract', 'volume_growth', 'approval_rate', 'market_segment']
                high_growth_display = high_growth[display_cols].copy()
                high_growth_display.columns = ['Census Tract', 'Growth Rate', 'Approval Rate', 'Segment']
                st.dataframe(high_growth_display, use_container_width=True)
        
        with col2:
            if not emerging.empty:
                st.markdown("**🌱 Emerging Markets**")
                display_cols = ['census_tract', 'volume_growth', 'approval_rate', 'market_segment']
                emerging_display = emerging[display_cols].copy()
                emerging_display.columns = ['Census Tract', 'Growth Rate', 'Approval Rate', 'Segment']
                st.dataframe(emerging_display, use_container_width=True)
    else:
        st.info("No specific growth opportunities identified")
    
    # Risk areas
    st.markdown("### ⚠️ Risk Areas")
    
    risks = business_insights['risk_alerts']
    
    if risks:
        risk_df = pd.DataFrame(risks)
        
        # Color code by risk level
        high_risk = risk_df[risk_df['risk_level'] == 'High']
        moderate_risk = risk_df[risk_df['risk_level'] == 'Moderate']
        
        if not high_risk.empty:
            st.markdown("**🚨 High Risk Areas**")
            display_cols = ['census_tract', 'volume_change', 'approval_rate_change', 'current_approval_rate', 'market_segment']
            high_risk_display = high_risk[display_cols].copy()
            high_risk_display.columns = ['Census Tract', 'Volume Change', 'Approval Change', 'Current Approval', 'Segment']
            st.dataframe(high_risk_display, use_container_width=True)
        
        if not moderate_risk.empty:
            st.markdown("**⚠️ Moderate Risk Areas**")
            display_cols = ['census_tract', 'volume_change', 'approval_rate_change', 'current_approval_rate', 'market_segment']
            moderate_risk_display = moderate_risk[display_cols].copy()
            moderate_risk_display.columns = ['Census Tract', 'Volume Change', 'Approval Change', 'Current Approval', 'Segment']
            st.dataframe(moderate_risk_display, use_container_width=True)
    else:
        st.info("No significant risk areas identified")
    
    # Strategic recommendations
    st.markdown("### 📋 Strategic Recommendations")
    
    recommendations = business_insights['strategic_recommendations']
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    else:
        st.info("No specific strategic recommendations available")
    
    # Market trends summary
    st.markdown("### 📈 Market Trends Summary")
    
    trends = business_insights['market_trends']
    
    if trends:
        trend_data = []
        for segment, trend_info in trends.items():
            trend_data.append({
                'Market Segment': segment,
                'Volume Trend': trend_info['volume_trend'],
                'Approval Trend': trend_info['approval_trend'],
                'Tract Count': trend_info['tract_count'],
                'Performance Level': trend_info['performance_level']
            })
        
        trend_df = pd.DataFrame(trend_data)
        st.dataframe(trend_df, use_container_width=True)
    else:
        st.info("No market trends data available")


# Integration function for main dashboard
def integrate_yoy_analysis_to_main_dashboard():
    """Integration point for adding YoY analysis to main executive dashboard"""
    
    # Add this as a new tab or section in executive_dashboard.py
    st.markdown("---")
    st.markdown("## 📈 Enhanced Year-over-Year Analysis")
    
    if st.button("🚀 Launch Comprehensive YoY Analysis Dashboard"):
        # Switch to dedicated YoY analysis page
        create_enhanced_yoy_dashboard()