#!/usr/bin/env python3
"""
Market Segmentation and Clustering System

Advanced market segmentation for lending opportunities using:
1. K-means clustering for market segments
2. Demographic and economic profiling
3. Risk-return matrix analysis
4. Product recommendation engine
5. Competitive positioning analysis

Provides strategic insights for market targeting and product development.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')

# Configuration
# Note: Logging is configured in the main pipeline module  
logger = logging.getLogger(__name__)


class MarketSegmenter:
    """Advanced market segmentation and clustering system"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.clusters = {}
        self.segment_profiles = {}
        self.optimal_k = None
        
    def prepare_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for market segmentation"""
        
        logger.info("Preparing features for market segmentation...")
        
        # Select key features for clustering
        clustering_features = [
            # Economic indicators
            'median_household_income', 'total_population', 'unemployment_rate',
            
            # Lending activity
            'loan_count', 'avg_loan_amount', 'approval_rate', 'denial_rate',
            
            # Opportunity components
            'opportunity_score', 'market_accessibility', 'risk_factors',
            'economic_indicators', 'lending_activity',
            
            # Additional demographic features
            'civilian_labor_force'
        ]
        
        # Filter available features
        available_features = [col for col in clustering_features if col in df.columns]
        
        # Create clustering dataset
        cluster_data = df[['census_tract'] + available_features].copy()
        
        # Handle missing values
        for col in available_features:
            cluster_data[col] = pd.to_numeric(cluster_data[col], errors='coerce')
            cluster_data[col] = cluster_data[col].fillna(cluster_data[col].median())
        
        # Create additional derived features
        if 'median_household_income' in cluster_data.columns and 'avg_loan_amount' in cluster_data.columns:
            cluster_data['loan_to_income_capacity'] = cluster_data['avg_loan_amount'] / (cluster_data['median_household_income'] + 1)
        
        if 'approval_rate' in cluster_data.columns and 'denial_rate' in cluster_data.columns:
            cluster_data['lending_efficiency'] = cluster_data['approval_rate'] / (cluster_data['denial_rate'] + 0.01)
        
        if 'total_population' in cluster_data.columns and 'loan_count' in cluster_data.columns:
            cluster_data['lending_penetration'] = cluster_data['loan_count'] / (cluster_data['total_population'] + 1) * 1000
        
        # Update feature list
        numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('census_tract') if 'census_tract' in numeric_cols else None
        
        logger.info(f"Clustering features prepared: {len(numeric_cols)} features")
        return cluster_data, numeric_cols
    
    def find_optimal_clusters(self, X: np.ndarray, max_k: int = 8) -> int:
        """Find optimal number of clusters using elbow method and silhouette analysis"""
        
        logger.info("Finding optimal number of clusters...")
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate inertia (within-cluster sum of squares)
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score
            sil_score = silhouette_score(X, cluster_labels)
            silhouette_scores.append(sil_score)
            
            logger.info(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")
        
        # Find optimal k using silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        logger.info(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    
    def perform_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform market segmentation clustering"""
        
        logger.info("Performing market segmentation clustering...")
        
        # Prepare features
        cluster_data, feature_cols = self.prepare_clustering_features(df)
        
        # Scale features
        X = cluster_data[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Find optimal number of clusters
        self.optimal_k = self.find_optimal_clusters(X_scaled, max_k=6)
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add cluster assignments to data
        result_df = df.copy()
        result_df['market_segment'] = cluster_labels
        result_df['segment_name'] = result_df['market_segment'].apply(self._get_segment_name)
        
        # Store clustering results
        self.clusters = {
            'model': kmeans,
            'scaler': self.scaler,
            'feature_columns': feature_cols,
            'cluster_centers': kmeans.cluster_centers_,
            'labels': cluster_labels
        }
        
        logger.info(f"Clustering completed: {self.optimal_k} segments identified")
        return result_df
    
    def _get_segment_name(self, segment_id: int) -> str:
        """Generate meaningful segment names"""
        
        segment_names = {
            0: "Prime Urban Markets",
            1: "Emerging Suburban Areas", 
            2: "High-Risk Rural Communities",
            3: "Stable Middle Markets",
            4: "Low-Activity Zones",
            5: "Premium Opportunity Areas"
        }
        
        return segment_names.get(segment_id, f"Segment {segment_id}")
    
    def create_segment_profiles(self, df: pd.DataFrame) -> Dict:
        """Create detailed profiles for each market segment"""
        
        logger.info("Creating market segment profiles...")
        
        profiles = {}
        
        for segment in df['market_segment'].unique():
            segment_data = df[df['market_segment'] == segment]
            segment_name = segment_data['segment_name'].iloc[0]
            
            # Convert segment to native Python int for JSON serialization
            segment_key = int(segment)
            
            # Basic statistics
            profile = {
                'segment_id': segment_key,
                'segment_name': str(segment_name),
                'tract_count': int(len(segment_data)),
                'percentage_of_market': float(len(segment_data) / len(df) * 100)
            }
            
            # Demographic profile
            demo_features = ['total_population', 'median_household_income', 'unemployment_rate', 'civilian_labor_force']
            for feature in demo_features:
                if feature in segment_data.columns:
                    profile[f'avg_{feature}'] = float(segment_data[feature].mean())
                    profile[f'median_{feature}'] = float(segment_data[feature].median())
            
            # Lending activity profile
            lending_features = ['loan_count', 'avg_loan_amount', 'approval_rate', 'denial_rate']
            for feature in lending_features:
                if feature in segment_data.columns:
                    profile[f'avg_{feature}'] = float(segment_data[feature].mean())
            
            # Opportunity profile
            opportunity_features = ['opportunity_score', 'market_accessibility', 'risk_factors', 
                                  'economic_indicators', 'lending_activity']
            for feature in opportunity_features:
                if feature in segment_data.columns:
                    profile[f'avg_{feature}'] = float(segment_data[feature].mean())
            
            # Risk-return characteristics
            profile['risk_level'] = self._assess_risk_level(segment_data)
            profile['return_potential'] = self._assess_return_potential(segment_data)
            profile['strategic_priority'] = self._determine_strategic_priority(profile)
            
            profiles[str(segment_key)] = profile  # Use string key for JSON serialization
        
        self.segment_profiles = profiles
        logger.info(f"Segment profiles created for {len(profiles)} segments")
        
        return profiles
    
    def _assess_risk_level(self, segment_data: pd.DataFrame) -> str:
        """Assess risk level for a market segment"""
        
        risk_indicators = []
        
        # Unemployment risk
        if 'unemployment_rate' in segment_data.columns:
            avg_unemployment = segment_data['unemployment_rate'].mean()
            if avg_unemployment > 8:
                risk_indicators.append('High unemployment')
            elif avg_unemployment > 5:
                risk_indicators.append('Moderate unemployment')
        
        # Denial rate risk
        if 'denial_rate' in segment_data.columns:
            avg_denial = segment_data['denial_rate'].mean()
            if avg_denial > 0.3:
                risk_indicators.append('High denial rate')
            elif avg_denial > 0.15:
                risk_indicators.append('Moderate denial rate')
        
        # Income stability
        if 'median_household_income' in segment_data.columns:
            avg_income = segment_data['median_household_income'].mean()
            if avg_income < 40000:
                risk_indicators.append('Low income levels')
            elif avg_income < 60000:
                risk_indicators.append('Moderate income levels')
        
        # Overall risk assessment
        if len(risk_indicators) >= 2:
            return 'High'
        elif len(risk_indicators) == 1:
            return 'Moderate'
        else:
            return 'Low'
    
    def _assess_return_potential(self, segment_data: pd.DataFrame) -> str:
        """Assess return potential for a market segment"""
        
        return_indicators = []
        
        # Market size
        if 'total_population' in segment_data.columns:
            avg_population = segment_data['total_population'].mean()
            if avg_population > 3000:
                return_indicators.append('Large market size')
        
        # Lending activity
        if 'loan_count' in segment_data.columns:
            avg_loans = segment_data['loan_count'].mean()
            if avg_loans > 50:
                return_indicators.append('High lending activity')
            elif avg_loans > 20:
                return_indicators.append('Moderate lending activity')
        
        # Income levels
        if 'median_household_income' in segment_data.columns:
            avg_income = segment_data['median_household_income'].mean()
            if avg_income > 70000:
                return_indicators.append('High income market')
        
        # Opportunity score
        if 'opportunity_score' in segment_data.columns:
            avg_score = segment_data['opportunity_score'].mean()
            if avg_score > 70:
                return_indicators.append('High opportunity score')
        
        # Overall return assessment
        if len(return_indicators) >= 3:
            return 'High'
        elif len(return_indicators) >= 2:
            return 'Moderate'
        else:
            return 'Low'
    
    def _determine_strategic_priority(self, profile: Dict) -> str:
        """Determine strategic priority for market segment"""
        
        risk = profile.get('risk_level', 'Moderate')
        return_potential = profile.get('return_potential', 'Moderate')
        
        # Strategic priority matrix
        priority_matrix = {
            ('Low', 'High'): 'High Priority',
            ('Low', 'Moderate'): 'High Priority',
            ('Moderate', 'High'): 'Medium Priority',
            ('Moderate', 'Moderate'): 'Medium Priority',
            ('High', 'High'): 'Medium Priority',
            ('Low', 'Low'): 'Low Priority',
            ('Moderate', 'Low'): 'Low Priority',
            ('High', 'Moderate'): 'Low Priority',
            ('High', 'Low'): 'Avoid'
        }
        
        return priority_matrix.get((risk, return_potential), 'Medium Priority')
    
    def generate_product_recommendations(self, df: pd.DataFrame) -> Dict:
        """Generate product recommendations for each market segment"""
        
        logger.info("Generating product recommendations...")
        
        recommendations = {}
        
        for segment in df['market_segment'].unique():
            segment_data = df[df['market_segment'] == segment]
            segment_key = str(int(segment))  # Convert to string for consistency
            segment_name = str(segment_data['segment_name'].iloc[0])
            
            # Analyze segment characteristics
            avg_income = segment_data['median_household_income'].mean()
            avg_loan_amount = segment_data.get('avg_loan_amount', {}).mean() if 'avg_loan_amount' in segment_data.columns else 0
            approval_rate = segment_data.get('approval_rate', {}).mean() if 'approval_rate' in segment_data.columns else 0
            risk_level = self.segment_profiles[segment_key]['risk_level']
            
            # Product recommendations based on segment profile
            products = []
            
            # Income-based products
            if avg_income > 80000:
                products.extend(['Jumbo Loans', 'Premium Mortgage Products', 'Investment Property Loans'])
            elif avg_income > 50000:
                products.extend(['Conventional Loans', 'FHA Loans', 'Refinancing Products'])
            else:
                products.extend(['FHA Loans', 'VA Loans', 'First-Time Buyer Programs'])
            
            # Risk-based products
            if risk_level == 'Low':
                products.extend(['Prime Rate Loans', 'Quick Approval Products'])
            elif risk_level == 'Moderate':
                products.extend(['Standard Products', 'Credit Enhancement Programs'])
            else:
                products.extend(['Assisted Purchase Programs', 'Credit Counseling Services'])
            
            # Market size considerations
            tract_count = len(segment_data)
            if tract_count > 5:
                products.append('Branch/Office Expansion')
            else:
                products.append('Digital/Mobile Services')
            
            recommendations[segment_key] = {  # Use string key directly
                'segment_name': segment_name,
                'primary_products': products[:3],
                'secondary_products': products[3:],
                'marketing_channels': self._recommend_marketing_channels(segment_data),
                'pricing_strategy': self._recommend_pricing_strategy(risk_level, approval_rate)
            }
        
        return recommendations
    
    def _recommend_marketing_channels(self, segment_data: pd.DataFrame) -> List[str]:
        """Recommend marketing channels for segment"""
        
        channels = []
        
        # Age-based (assuming younger populations are more digital)
        avg_income = segment_data['median_household_income'].mean()
        population = segment_data['total_population'].mean()
        
        if avg_income > 70000:
            channels.extend(['Digital Marketing', 'Professional Networks', 'Direct Mail'])
        else:
            channels.extend(['Community Outreach', 'Local Partnerships', 'Social Media'])
        
        if population > 2000:
            channels.append('Local Advertising')
        else:
            channels.append('Referral Programs')
        
        return channels[:3]
    
    def _recommend_pricing_strategy(self, risk_level: str, approval_rate: float) -> str:
        """Recommend pricing strategy for segment"""
        
        if risk_level == 'Low' and approval_rate > 0.8:
            return 'Competitive Pricing'
        elif risk_level == 'Moderate':
            return 'Standard Pricing'
        else:
            return 'Risk-Adjusted Pricing'
    
    def create_risk_return_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk-return matrix for market segments"""
        
        logger.info("Creating risk-return matrix...")
        
        matrix_data = []
        
        for segment in df['market_segment'].unique():
            segment_data = df[df['market_segment'] == segment]
            segment_key = str(int(segment))  # Convert to string for consistency
            profile = self.segment_profiles[segment_key]
            
            # Calculate risk score (0-100, higher = more risky)
            risk_score = 0
            if 'avg_unemployment_rate' in profile:
                risk_score += min(profile['avg_unemployment_rate'] * 10, 50)
            if 'avg_denial_rate' in profile:
                risk_score += profile['avg_denial_rate'] * 100
            risk_score = min(risk_score, 100)
            
            # Calculate return score (opportunity score)
            return_score = profile.get('avg_opportunity_score', 50)
            
            matrix_data.append({
                'segment_id': int(segment_key),
                'segment_name': str(profile['segment_name']),
                'risk_score': float(risk_score),
                'return_score': float(return_score),
                'tract_count': int(profile['tract_count']),
                'strategic_priority': str(profile['strategic_priority'])
            })
        
        return pd.DataFrame(matrix_data)
    
    def generate_competitive_analysis(self, df: pd.DataFrame) -> Dict:
        """Generate competitive positioning analysis"""
        
        logger.info("Generating competitive analysis...")
        
        analysis = {}
        
        # Market share opportunities
        total_loans = df['loan_count'].sum() if 'loan_count' in df.columns else 0
        total_market_value = df['avg_loan_amount'].sum() if 'avg_loan_amount' in df.columns else 0
        
        for segment in df['market_segment'].unique():
            segment_data = df[df['market_segment'] == segment]
            segment_key = str(int(segment))  # Convert to string for consistency
            profile = self.segment_profiles[segment_key]
            
            segment_loans = segment_data['loan_count'].sum() if 'loan_count' in segment_data.columns else 0
            segment_value = segment_data['avg_loan_amount'].sum() if 'avg_loan_amount' in segment_data.columns else 0
            
            analysis[segment_key] = {  # Use string key directly
                'segment_name': str(profile['segment_name']),
                'market_share_potential': float(segment_loans / total_loans * 100) if total_loans > 0 else 0.0,
                'value_share_potential': float(segment_value / total_market_value * 100) if total_market_value > 0 else 0.0,
                'competitive_intensity': str(self._assess_competitive_intensity(segment_data)),
                'market_attractiveness': str(self._assess_market_attractiveness(profile)),
                'recommended_strategy': str(self._recommend_competitive_strategy(profile))
            }
        
        return analysis
    
    def _assess_competitive_intensity(self, segment_data: pd.DataFrame) -> str:
        """Assess competitive intensity in segment"""
        
        # Use approval rate as proxy for competition
        if 'approval_rate' in segment_data.columns:
            avg_approval = segment_data['approval_rate'].mean()
            if avg_approval > 0.8:
                return 'High'
            elif avg_approval > 0.6:
                return 'Moderate'
            else:
                return 'Low'
        
        return 'Moderate'
    
    def _assess_market_attractiveness(self, profile: Dict) -> str:
        """Assess overall market attractiveness"""
        
        attractiveness_score = 0
        
        # Size factor
        if profile['tract_count'] > 5:
            attractiveness_score += 2
        elif profile['tract_count'] > 2:
            attractiveness_score += 1
        
        # Return potential
        if profile['return_potential'] == 'High':
            attractiveness_score += 3
        elif profile['return_potential'] == 'Moderate':
            attractiveness_score += 2
        else:
            attractiveness_score += 1
        
        # Risk factor (inverse)
        if profile['risk_level'] == 'Low':
            attractiveness_score += 2
        elif profile['risk_level'] == 'Moderate':
            attractiveness_score += 1
        
        if attractiveness_score >= 6:
            return 'High'
        elif attractiveness_score >= 4:
            return 'Moderate'
        else:
            return 'Low'
    
    def _recommend_competitive_strategy(self, profile: Dict) -> str:
        """Recommend competitive strategy for segment"""
        
        priority = profile['strategic_priority']
        risk = profile['risk_level']
        return_potential = profile['return_potential']
        
        if priority == 'High Priority':
            return 'Aggressive Market Entry'
        elif priority == 'Medium Priority' and risk == 'Low':
            return 'Selective Market Entry'
        elif priority == 'Medium Priority':
            return 'Partnership Strategy'
        elif priority == 'Low Priority':
            return 'Monitor and Evaluate'
        else:
            return 'Avoid Market'


def perform_comprehensive_segmentation(df: pd.DataFrame, config) -> Tuple[pd.DataFrame, Dict]:
    """Perform comprehensive market segmentation analysis"""
    
    logger.info("Performing comprehensive market segmentation...")
    
    # Initialize segmenter
    segmenter = MarketSegmenter(config)
    
    # Perform clustering
    segmented_data = segmenter.perform_clustering(df)
    
    # Create segment profiles
    segment_profiles = segmenter.create_segment_profiles(segmented_data)
    
    # Generate recommendations
    product_recommendations = segmenter.generate_product_recommendations(segmented_data)
    
    # Create risk-return matrix
    risk_return_matrix = segmenter.create_risk_return_matrix(segmented_data)
    
    # Generate competitive analysis
    competitive_analysis = segmenter.generate_competitive_analysis(segmented_data)
    
    # Compile comprehensive results
    analysis_results = {
        'segment_profiles': segment_profiles,
        'product_recommendations': product_recommendations,
        'risk_return_matrix': risk_return_matrix.to_dict('records'),
        'competitive_analysis': competitive_analysis,
        'clustering_info': {
            'optimal_clusters': segmenter.optimal_k,
            'total_tracts': len(segmented_data),
            'features_used': len(segmenter.clusters['feature_columns'])
        }
    }
    
    return segmented_data, analysis_results


if __name__ == "__main__":
    from advanced_lending_platform import LendingConfig, DataProcessor, OpportunityScoreCalculator
    
    # Initialize configuration
    config = LendingConfig()
    
    # Load and process data
    processor = DataProcessor(config)
    data_sources = processor.load_all_data()
    master_data = processor.create_master_dataset()
    
    # Calculate opportunity scores
    calculator = OpportunityScoreCalculator(config)
    scored_data = calculator.calculate_opportunity_score(master_data)
    
    # Perform segmentation
    segmented_data, analysis_results = perform_comprehensive_segmentation(scored_data, config)
    
    logger.info("Segmentation analysis completed successfully")
    
    # Display summary
    print("\n" + "="*60)
    print("MARKET SEGMENTATION ANALYSIS - SUMMARY")
    print("="*60)
    print(f"Total Market Segments: {analysis_results['clustering_info']['optimal_clusters']}")
    print(f"Census Tracts Analyzed: {analysis_results['clustering_info']['total_tracts']}")
    
    print("\nSegment Distribution:")
    segment_counts = segmented_data['segment_name'].value_counts()
    for segment, count in segment_counts.items():
        print(f"  {segment}: {count} tracts")
    
    print("\nStrategic Priorities:")
    for segment_id, profile in analysis_results['segment_profiles'].items():
        print(f"  {profile['segment_name']}: {profile['strategic_priority']}")