#!/usr/bin/env python3
"""
Enhanced Year-over-Year Analysis Runner

Simple script to run the enhanced year-over-year census tract analysis
independently or as part of the main dashboard.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
from datetime import datetime

# Add the src directory to Python path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

try:
    from enhanced_yoy_dashboard import create_enhanced_yoy_dashboard
    
    def main():
        """Run the enhanced year-over-year analysis dashboard"""
        
        st.set_page_config(
            page_title="Year-over-Year Census Tract Analysis",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
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
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown("""
        # 📈 Enhanced Year-over-Year Census Tract Analysis
        
        **Comprehensive multi-year performance analysis for Kansas HMDA lending data**
        
        This analysis provides detailed insights into how census tracts are performing year-over-year,
        identifying growth opportunities, market trends, and strategic insights for lending decisions.
        """)
        
        # Navigation sidebar
        st.sidebar.title("📊 Analysis Navigation")
        
        run_mode = st.sidebar.radio(
            "Select Analysis Mode:",
            ["Full Dashboard", "Quick Summary", "Data Export"]
        )
        
        if run_mode == "Full Dashboard":
            # Run the complete enhanced dashboard
            create_enhanced_yoy_dashboard()
            
        elif run_mode == "Quick Summary":
            # Run just the summary analysis
            st.subheader("🚀 Quick Summary Analysis")
            
            try:
                from enhanced_yoy_analyzer import YearOverYearAnalyzer
                
                analyzer = YearOverYearAnalyzer()
                
                with st.spinner("Loading and analyzing multi-year data..."):
                    analyzer.load_multi_year_data()
                    yoy_analysis = analyzer.perform_yoy_analysis()
                    business_insights = analyzer.generate_business_insights()
                
                # Display executive summary
                exec_summary = business_insights['executive_summary']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Analysis Period", exec_summary['analysis_period'])
                with col2:
                    st.metric("Census Tracts", f"{exec_summary['total_tracts_analyzed']:,}")
                with col3:
                    st.metric("Volume Trend", f"{exec_summary['volume_trend']:+.1f}%")
                with col4:
                    st.metric("Approval Trend", f"{exec_summary['approval_rate_trend']:+.1f}%")
                
                # Show key insights
                st.subheader("💡 Key Strategic Insights")
                recommendations = business_insights['strategic_recommendations']
                for i, rec in enumerate(recommendations[:5], 1):
                    st.markdown(f"{i}. {rec}")
                
                # Show top opportunities
                opportunities = business_insights['growth_opportunities']
                if opportunities:
                    st.subheader("🏆 Top Growth Opportunities")
                    for opp in opportunities[:3]:
                        st.markdown(f"**{opp['census_tract']}**: {opp['type']} - {opp['volume_growth']} growth, {opp['approval_rate']} approval rate")
                
            except Exception as e:
                st.error(f"Error in quick analysis: {str(e)}")
                
        elif run_mode == "Data Export":
            # Data export functionality
            st.subheader("📥 Data Export")
            
            try:
                from enhanced_yoy_analyzer import YearOverYearAnalyzer
                import json
                
                analyzer = YearOverYearAnalyzer()
                
                with st.spinner("Preparing data for export..."):
                    analyzer.load_multi_year_data()
                    yoy_analysis = analyzer.perform_yoy_analysis()
                    business_insights = analyzer.generate_business_insights()
                
                # Export options
                export_format = st.selectbox(
                    "Select Export Format:",
                    ["JSON", "CSV Summary", "Excel Report"]
                )
                
                if st.button("📥 Generate Export"):
                    if export_format == "JSON":
                        # Full analysis export
                        export_data = {
                            'analysis_results': yoy_analysis,
                            'business_insights': business_insights,
                            'export_timestamp': str(datetime.now())
                        }
                        
                        json_str = json.dumps(export_data, indent=2, default=str)
                        st.download_button(
                            label="Download JSON Analysis",
                            data=json_str,
                            file_name=f"yoy_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                        
                    elif export_format == "CSV Summary":
                        # Create summary CSV
                        summary_data = []
                        
                        for year, metrics in yoy_analysis['yearly_metrics'].items():
                            summary_stats = metrics.groupby('market_segment').agg({
                                'total_applications': 'sum',
                                'approval_rate': 'mean',
                                'avg_loan_amount': 'mean'
                            }).round(2)
                            
                            for segment, stats in summary_stats.iterrows():
                                summary_data.append({
                                    'year': year,
                                    'market_segment': segment,
                                    'total_applications': stats['total_applications'],
                                    'approval_rate': stats['approval_rate'],
                                    'avg_loan_amount': stats['avg_loan_amount']
                                })
                        
                        summary_df = pd.DataFrame(summary_data)
                        csv_str = summary_df.to_csv(index=False)
                        
                        st.download_button(
                            label="Download CSV Summary",
                            data=csv_str,
                            file_name=f"yoy_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
            except Exception as e:
                st.error(f"Error in data export: {str(e)}")
        
        # Sidebar information
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ℹ️ About This Analysis")
        st.sidebar.markdown("""
        **Data Sources:**
        - HMDA 2022: 109,755 records
        - HMDA 2023: 85,831 records  
        - HMDA 2024: 94,531 records
        
        **Analysis Features:**
        - Year-over-year trend analysis
        - Market segment performance
        - Growth opportunity identification
        - Risk area detection
        - Strategic recommendations
        """)
        
        st.sidebar.markdown("### 🔄 Data Refresh")
        if st.sidebar.button("Refresh Analysis Data"):
            st.cache_data.clear()
            st.success("Cache cleared! Please rerun analysis.")
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all required modules are installed:")
    st.code("""
    pip install streamlit pandas numpy plotly scikit-learn
    """)