#!/usr/bin/env python3
"""
Enhanced Year-over-Year Census Tract Performance Analyzer

Provides comprehensive analysis of how census tracts perform across multiple years
with detailed business insights for lending opportunity identification.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class YearOverYearAnalyzer:
    """Enhanced year-over-year performance analysis for census tracts"""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/actual")
        self.yearly_data = {}
        self.yoy_analysis = {}
        self.market_segments = {}
        
    def load_multi_year_data(self) -> Dict[int, pd.DataFrame]:
        """Load HMDA data for multiple years"""
        logger.info("Loading multi-year HMDA data for YoY analysis...")
        
        yearly_files = {
            2022: self.data_dir / "2022_state_KS.csv",
            2023: self.data_dir / "2023_state_KS.csv", 
            2024: self.data_dir / "2024_state_KS.csv"
        }
        
        for year, file_path in yearly_files.items():
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    df['year'] = year
                    self.yearly_data[year] = df
                    logger.info(f"Loaded {len(df):,} records for {year}")
                except Exception as e:
                    logger.error(f"Error loading {year} data: {str(e)}")
        
        return self.yearly_data
    
    def calculate_tract_performance_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive performance metrics for each census tract"""
        
        tract_metrics = []
        
        for tract in df['census_tract'].unique():
            if pd.isna(tract):
                continue
                
            tract_data = df[df['census_tract'] == tract].copy()
            
            # Basic volume metrics
            total_applications = len(tract_data)
            total_originations = len(tract_data[tract_data['action_taken'] == 1])
            total_denials = len(tract_data[tract_data['action_taken'] == 3])
            total_withdrawals = len(tract_data[tract_data['action_taken'] == 4])
            
            # Performance ratios
            approval_rate = (total_originations / total_applications * 100) if total_applications > 0 else 0
            denial_rate = (total_denials / total_applications * 100) if total_applications > 0 else 0
            withdrawal_rate = (total_withdrawals / total_applications * 100) if total_applications > 0 else 0
            
            # Financial metrics
            avg_loan_amount = tract_data['loan_amount'].mean() if 'loan_amount' in tract_data.columns else 0
            median_loan_amount = tract_data['loan_amount'].median() if 'loan_amount' in tract_data.columns else 0
            total_loan_volume = tract_data[tract_data['action_taken'] == 1]['loan_amount'].sum() if 'loan_amount' in tract_data.columns else 0
            
            # Income and DTI metrics
            avg_income = tract_data['income'].mean() if 'income' in tract_data.columns else 0
            median_income = tract_data['income'].median() if 'income' in tract_data.columns else 0
            
            # DTI analysis (convert string percentages to numeric)
            if 'debt_to_income_ratio' in tract_data.columns:
                dti_numeric = pd.to_numeric(tract_data['debt_to_income_ratio'].astype(str).str.replace('%', '').str.replace('>', '').str.replace('<', ''), errors='coerce')
                avg_dti = dti_numeric.mean()
                median_dti = dti_numeric.median()
            else:
                avg_dti = 0
                median_dti = 0
            
            # Interest rate metrics
            if 'interest_rate' in tract_data.columns:
                avg_interest_rate = pd.to_numeric(tract_data['interest_rate'], errors='coerce').mean()
                median_interest_rate = pd.to_numeric(tract_data['interest_rate'], errors='coerce').median()
            else:
                avg_interest_rate = 0
                median_interest_rate = 0
            
            # Loan type distribution
            conventional_pct = len(tract_data[tract_data['loan_type'] == 1]) / total_applications * 100 if total_applications > 0 else 0
            fha_pct = len(tract_data[tract_data['loan_type'] == 2]) / total_applications * 100 if total_applications > 0 else 0
            va_pct = len(tract_data[tract_data['loan_type'] == 3]) / total_applications * 100 if total_applications > 0 else 0
            
            # Property value metrics
            if 'property_value' in tract_data.columns:
                avg_property_value = pd.to_numeric(tract_data['property_value'], errors='coerce').mean()
                median_property_value = pd.to_numeric(tract_data['property_value'], errors='coerce').median()
            else:
                avg_property_value = 0
                median_property_value = 0
            
            tract_metrics.append({
                'census_tract': tract,
                'year': df['year'].iloc[0],
                
                # Volume metrics
                'total_applications': total_applications,
                'total_originations': total_originations,
                'total_denials': total_denials,
                'total_withdrawals': total_withdrawals,
                
                # Performance rates
                'approval_rate': approval_rate,
                'denial_rate': denial_rate,
                'withdrawal_rate': withdrawal_rate,
                
                # Financial metrics
                'avg_loan_amount': avg_loan_amount,
                'median_loan_amount': median_loan_amount,
                'total_loan_volume': total_loan_volume,
                'avg_income': avg_income,
                'median_income': median_income,
                'avg_dti': avg_dti,
                'median_dti': median_dti,
                'avg_interest_rate': avg_interest_rate,
                'median_interest_rate': median_interest_rate,
                'avg_property_value': avg_property_value,
                'median_property_value': median_property_value,
                
                # Product mix
                'conventional_pct': conventional_pct,
                'fha_pct': fha_pct,
                'va_pct': va_pct,
                
                # Market segment indicators
                'market_segment': self._classify_market_segment(avg_loan_amount, approval_rate, total_applications),
                'performance_tier': self._classify_performance_tier(approval_rate, total_applications)
            })
        
        return pd.DataFrame(tract_metrics)
    
    def _classify_market_segment(self, avg_loan_amount: float, approval_rate: float, volume: int) -> str:
        """Classify market segment based on loan characteristics"""
        
        if avg_loan_amount >= 400000:
            return "Luxury"
        elif avg_loan_amount >= 250000:
            return "Premium" 
        elif avg_loan_amount >= 150000:
            return "Mainstream"
        elif avg_loan_amount >= 100000:
            return "Value"
        else:
            return "Affordable"
    
    def _classify_performance_tier(self, approval_rate: float, volume: int) -> str:
        """Classify performance tier based on approval rate and volume"""
        
        if approval_rate >= 70 and volume >= 50:
            return "High Performer"
        elif approval_rate >= 60 and volume >= 30:
            return "Strong Performer"
        elif approval_rate >= 50 and volume >= 15:
            return "Moderate Performer"
        elif approval_rate >= 40:
            return "Challenging Market"
        else:
            return "Difficult Market"
    
    def perform_yoy_analysis(self) -> Dict:
        """Perform comprehensive year-over-year analysis"""
        
        logger.info("Performing comprehensive year-over-year analysis...")
        
        if not self.yearly_data:
            self.load_multi_year_data()
        
        # Calculate metrics for each year
        yearly_metrics = {}
        for year, df in self.yearly_data.items():
            yearly_metrics[year] = self.calculate_tract_performance_metrics(df)
        
        # Combine all years for YoY comparison
        all_metrics = pd.concat(yearly_metrics.values(), ignore_index=True)
        
        # Calculate year-over-year changes
        yoy_changes = self._calculate_yoy_changes(all_metrics)
        
        # Market segment analysis
        segment_analysis = self._analyze_market_segments(all_metrics)
        
        # Performance trend analysis
        trend_analysis = self._analyze_performance_trends(all_metrics)
        
        # Top performers and decliners
        performance_rankings = self._rank_performance_changes(yoy_changes)
        
        self.yoy_analysis = {
            'yearly_metrics': yearly_metrics,
            'yoy_changes': yoy_changes,
            'segment_analysis': segment_analysis,
            'trend_analysis': trend_analysis,
            'performance_rankings': performance_rankings,
            'summary_stats': self._calculate_summary_statistics(all_metrics)
        }
        
        return self.yoy_analysis
    
    def _calculate_yoy_changes(self, all_metrics: pd.DataFrame) -> pd.DataFrame:
        """Calculate year-over-year changes for all metrics"""
        
        yoy_results = []
        
        # Get all unique census tracts
        tracts = all_metrics['census_tract'].unique()
        years = sorted(all_metrics['year'].unique())
        
        for tract in tracts:
            tract_data = all_metrics[all_metrics['census_tract'] == tract].sort_values('year')
            
            if len(tract_data) < 2:
                continue  # Need at least 2 years for comparison
            
            # Calculate YoY changes for each consecutive year pair
            for i in range(1, len(tract_data)):
                current_year = tract_data.iloc[i]
                previous_year = tract_data.iloc[i-1]
                
                # Calculate percentage changes
                volume_change = ((current_year['total_applications'] - previous_year['total_applications']) / 
                               previous_year['total_applications'] * 100) if previous_year['total_applications'] > 0 else 0
                
                approval_rate_change = current_year['approval_rate'] - previous_year['approval_rate']
                
                loan_amount_change = ((current_year['avg_loan_amount'] - previous_year['avg_loan_amount']) / 
                                    previous_year['avg_loan_amount'] * 100) if previous_year['avg_loan_amount'] > 0 else 0
                
                income_change = ((current_year['avg_income'] - previous_year['avg_income']) / 
                               previous_year['avg_income'] * 100) if previous_year['avg_income'] > 0 else 0
                
                yoy_results.append({
                    'census_tract': tract,
                    'from_year': int(previous_year['year']),
                    'to_year': int(current_year['year']),
                    'volume_change_pct': volume_change,
                    'approval_rate_change': approval_rate_change,
                    'loan_amount_change_pct': loan_amount_change,
                    'income_change_pct': income_change,
                    'current_volume': current_year['total_applications'],
                    'current_approval_rate': current_year['approval_rate'],
                    'current_avg_loan': current_year['avg_loan_amount'],
                    'market_segment': current_year['market_segment'],
                    'performance_tier': current_year['performance_tier']
                })
        
        return pd.DataFrame(yoy_results)
    
    def _analyze_market_segments(self, all_metrics: pd.DataFrame) -> Dict:
        """Analyze performance by market segment over time"""
        
        segment_analysis = {}
        
        for segment in all_metrics['market_segment'].unique():
            if pd.isna(segment):
                continue
                
            segment_data = all_metrics[all_metrics['market_segment'] == segment]
            
            # Calculate segment performance by year
            yearly_segment_performance = segment_data.groupby('year').agg({
                'total_applications': 'sum',
                'approval_rate': 'mean',
                'avg_loan_amount': 'mean',
                'total_loan_volume': 'sum'
            }).round(2)
            
            # Calculate segment trends
            if len(yearly_segment_performance) > 1:
                volume_trend = yearly_segment_performance['total_applications'].pct_change().mean() * 100
                approval_trend = yearly_segment_performance['approval_rate'].diff().mean()
            else:
                volume_trend = 0
                approval_trend = 0
            
            segment_analysis[segment] = {
                'yearly_performance': yearly_segment_performance,
                'volume_trend_pct': volume_trend,
                'approval_rate_trend': approval_trend,
                'tract_count': len(segment_data['census_tract'].unique()),
                'avg_performance_tier': segment_data['performance_tier'].mode().iloc[0] if not segment_data['performance_tier'].empty else 'Unknown'
            }
        
        return segment_analysis
    
    def _analyze_performance_trends(self, all_metrics: pd.DataFrame) -> Dict:
        """Analyze overall performance trends across all tracts"""
        
        # Aggregate performance by year
        yearly_performance = all_metrics.groupby('year').agg({
            'total_applications': 'sum',
            'total_originations': 'sum', 
            'approval_rate': 'mean',
            'avg_loan_amount': 'mean',
            'avg_income': 'mean',
            'avg_dti': 'mean',
            'total_loan_volume': 'sum'
        }).round(2)
        
        # Calculate overall trends
        trends = {}
        for metric in yearly_performance.columns:
            if len(yearly_performance) > 1:
                if metric in ['total_applications', 'total_originations', 'total_loan_volume']:
                    # Volume metrics - calculate percentage change
                    trends[metric] = yearly_performance[metric].pct_change().mean() * 100
                else:
                    # Rate/average metrics - calculate absolute change
                    trends[metric] = yearly_performance[metric].diff().mean()
            else:
                trends[metric] = 0
        
        return {
            'yearly_performance': yearly_performance,
            'trends': trends,
            'total_years': len(yearly_performance),
            'top_growth_tracts': self._identify_growth_leaders(all_metrics),
            'emerging_markets': self._identify_emerging_markets(all_metrics)
        }
    
    def _identify_growth_leaders(self, all_metrics: pd.DataFrame) -> List[Dict]:
        """Identify census tracts with strongest growth"""
        
        growth_leaders = []
        
        for tract in all_metrics['census_tract'].unique():
            tract_data = all_metrics[all_metrics['census_tract'] == tract].sort_values('year')
            
            if len(tract_data) >= 2:
                # Calculate growth metrics
                first_year = tract_data.iloc[0]
                last_year = tract_data.iloc[-1]
                
                volume_growth = ((last_year['total_applications'] - first_year['total_applications']) / 
                               first_year['total_applications'] * 100) if first_year['total_applications'] > 0 else 0
                
                approval_improvement = last_year['approval_rate'] - first_year['approval_rate']
                
                if volume_growth > 20 and approval_improvement > 5:  # Strong growth criteria
                    growth_leaders.append({
                        'census_tract': tract,
                        'volume_growth_pct': volume_growth,
                        'approval_improvement': approval_improvement,
                        'current_volume': last_year['total_applications'],
                        'current_approval_rate': last_year['approval_rate'],
                        'market_segment': last_year['market_segment']
                    })
        
        return sorted(growth_leaders, key=lambda x: x['volume_growth_pct'], reverse=True)[:10]
    
    def _identify_emerging_markets(self, all_metrics: pd.DataFrame) -> List[Dict]:
        """Identify emerging market opportunities"""
        
        # Markets with increasing volume but still moderate competition
        emerging = []
        
        for tract in all_metrics['census_tract'].unique():
            tract_data = all_metrics[all_metrics['census_tract'] == tract].sort_values('year')
            
            if len(tract_data) >= 2:
                latest = tract_data.iloc[-1]
                
                # Criteria: Growing volume, decent approval rate, not oversaturated
                if (latest['total_applications'] >= 20 and 
                    latest['approval_rate'] >= 50 and
                    latest['total_applications'] <= 100):  # Not oversaturated
                    
                    # Check for growth trend
                    first_year = tract_data.iloc[0]
                    volume_growth = ((latest['total_applications'] - first_year['total_applications']) / 
                                   first_year['total_applications'] * 100) if first_year['total_applications'] > 0 else 0
                    
                    if volume_growth > 0:  # Any positive growth
                        emerging.append({
                            'census_tract': tract,
                            'current_volume': latest['total_applications'],
                            'current_approval_rate': latest['approval_rate'],
                            'volume_growth_pct': volume_growth,
                            'market_segment': latest['market_segment'],
                            'avg_loan_amount': latest['avg_loan_amount']
                        })
        
        return sorted(emerging, key=lambda x: x['volume_growth_pct'], reverse=True)[:15]
    
    def _rank_performance_changes(self, yoy_changes: pd.DataFrame) -> Dict:
        """Rank census tracts by performance changes"""
        
        if yoy_changes.empty:
            return {'top_improvers': [], 'top_decliners': []}
        
        # Get most recent year-over-year changes
        latest_changes = yoy_changes[yoy_changes['to_year'] == yoy_changes['to_year'].max()]
        
        # Top improvers (combination of volume and approval rate improvement)
        latest_changes['improvement_score'] = (
            latest_changes['volume_change_pct'] * 0.4 +
            latest_changes['approval_rate_change'] * 0.6
        )
        
        top_improvers = latest_changes.nlargest(10, 'improvement_score')[
            ['census_tract', 'volume_change_pct', 'approval_rate_change', 'current_volume', 
             'current_approval_rate', 'market_segment']
        ].to_dict('records')
        
        top_decliners = latest_changes.nsmallest(10, 'improvement_score')[
            ['census_tract', 'volume_change_pct', 'approval_rate_change', 'current_volume', 
             'current_approval_rate', 'market_segment']
        ].to_dict('records')
        
        return {
            'top_improvers': top_improvers,
            'top_decliners': top_decliners
        }
    
    def _calculate_summary_statistics(self, all_metrics: pd.DataFrame) -> Dict:
        """Calculate summary statistics across all years"""
        
        return {
            'total_census_tracts': len(all_metrics['census_tract'].unique()),
            'years_analyzed': sorted(all_metrics['year'].unique()),
            'total_applications_all_years': all_metrics['total_applications'].sum(),
            'avg_approval_rate_all_years': all_metrics['approval_rate'].mean(),
            'market_segment_distribution': all_metrics['market_segment'].value_counts().to_dict(),
            'performance_tier_distribution': all_metrics['performance_tier'].value_counts().to_dict()
        }

    def generate_business_insights(self) -> Dict:
        """Generate actionable business insights from YoY analysis"""
        
        if not self.yoy_analysis:
            self.perform_yoy_analysis()
        
        insights = {
            'executive_summary': self._generate_executive_summary(),
            'growth_opportunities': self._identify_growth_opportunities(),
            'risk_alerts': self._identify_risk_areas(),
            'market_trends': self._summarize_market_trends(),
            'strategic_recommendations': self._generate_strategic_recommendations()
        }
        
        return insights
    
    def _generate_executive_summary(self) -> Dict:
        """Generate executive summary of YoY performance"""
        
        summary_stats = self.yoy_analysis['summary_stats']
        trend_analysis = self.yoy_analysis['trend_analysis']
        
        years = summary_stats['years_analyzed']
        total_volume = summary_stats['total_applications_all_years']
        avg_approval = summary_stats['avg_approval_rate_all_years']
        
        return {
            'analysis_period': f"{min(years)}-{max(years)}",
            'total_tracts_analyzed': summary_stats['total_census_tracts'],
            'total_loan_applications': total_volume,
            'overall_approval_rate': round(avg_approval, 1),
            'volume_trend': round(trend_analysis['trends']['total_applications'], 1),
            'approval_rate_trend': round(trend_analysis['trends']['approval_rate'], 2),
            'dominant_market_segment': max(summary_stats['market_segment_distribution'], 
                                         key=summary_stats['market_segment_distribution'].get)
        }
    
    def _identify_growth_opportunities(self) -> List[Dict]:
        """Identify top growth opportunities"""
        
        opportunities = []
        
        # High-growth, high-approval markets
        growth_leaders = self.yoy_analysis['trend_analysis']['top_growth_tracts']
        emerging_markets = self.yoy_analysis['trend_analysis']['emerging_markets']
        
        for tract_info in growth_leaders[:5]:
            opportunities.append({
                'type': 'High Growth Leader',
                'census_tract': tract_info['census_tract'],
                'volume_growth': f"{tract_info['volume_growth_pct']:.1f}%",
                'approval_rate': f"{tract_info['current_approval_rate']:.1f}%",
                'market_segment': tract_info['market_segment'],
                'opportunity_score': 'High'
            })
        
        for tract_info in emerging_markets[:3]:
            opportunities.append({
                'type': 'Emerging Market',
                'census_tract': tract_info['census_tract'],
                'volume_growth': f"{tract_info['volume_growth_pct']:.1f}%",
                'approval_rate': f"{tract_info['current_approval_rate']:.1f}%",
                'market_segment': tract_info['market_segment'],
                'opportunity_score': 'Moderate'
            })
        
        return opportunities
    
    def _identify_risk_areas(self) -> List[Dict]:
        """Identify areas of concern requiring attention"""
        
        risks = []
        
        # Markets with declining performance
        decliners = self.yoy_analysis['performance_rankings']['top_decliners']
        
        for tract_info in decliners[:5]:
            risk_level = 'High' if tract_info['approval_rate_change'] < -10 else 'Moderate'
            
            risks.append({
                'type': 'Declining Performance',
                'census_tract': tract_info['census_tract'],
                'volume_change': f"{tract_info['volume_change_pct']:.1f}%",
                'approval_rate_change': f"{tract_info['approval_rate_change']:.1f}%",
                'current_approval_rate': f"{tract_info['current_approval_rate']:.1f}%",
                'market_segment': tract_info['market_segment'],
                'risk_level': risk_level
            })
        
        return risks
    
    def _summarize_market_trends(self) -> Dict:
        """Summarize overall market trends"""
        
        segment_analysis = self.yoy_analysis['segment_analysis']
        
        trends = {}
        for segment, data in segment_analysis.items():
            trends[segment] = {
                'volume_trend': f"{data['volume_trend_pct']:.1f}%",
                'approval_trend': f"{data['approval_rate_trend']:.1f}%",
                'tract_count': data['tract_count'],
                'performance_level': data['avg_performance_tier']
            }
        
        return trends
    
    def _generate_strategic_recommendations(self) -> List[str]:
        """Generate strategic recommendations based on analysis"""
        
        recommendations = []
        
        # Analyze trends for recommendations
        trend_analysis = self.yoy_analysis['trend_analysis']
        volume_trend = trend_analysis['trends']['total_applications']
        approval_trend = trend_analysis['trends']['approval_rate']
        
        if volume_trend > 10:
            recommendations.append("📈 Market is experiencing strong growth - consider expanding origination capacity")
        elif volume_trend < -10:
            recommendations.append("📉 Market volume declining - investigate competitive pressures and rate sensitivity")
        
        if approval_trend < -2:
            recommendations.append("⚠️ Approval rates declining - review underwriting criteria and credit policies")
        elif approval_trend > 2:
            recommendations.append("✅ Approval rates improving - good sign of market health and process optimization")
        
        # Segment-specific recommendations
        segment_analysis = self.yoy_analysis['segment_analysis']
        luxury_performance = segment_analysis.get('Luxury', {})
        if luxury_performance and luxury_performance.get('volume_trend_pct', 0) > 15:
            recommendations.append("💎 Luxury segment showing strong growth - consider premium product offerings")
        
        affordable_performance = segment_analysis.get('Affordable', {})
        if affordable_performance and affordable_performance.get('approval_rate_trend', 0) < -5:
            recommendations.append("🏠 Affordable segment approval rates declining - review FHA/VA programs")
        
        # Growth opportunity recommendations
        growth_leaders = self.yoy_analysis['trend_analysis']['top_growth_tracts']
        if len(growth_leaders) > 5:
            recommendations.append(f"🎯 {len(growth_leaders)} high-growth census tracts identified - prioritize these for business development")
        
        return recommendations