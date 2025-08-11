#!/usr/bin/env python3
"""
Bitcoin Market Sentiment & Hyperliquid Trader Behavior Analysis
Comprehensive data science analysis exploring the relationship between trader performance and market sentiment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from datetime import datetime, timedelta
import pickle
import os

warnings.filterwarnings('ignore')

# Set style for professional visualizations
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

class CryptoTradingAnalyzer:
    """
    Comprehensive cryptocurrency trading behavior analyzer
    """
    
    def __init__(self, data_path='../hyperliquid-sentiment-analysis/data'):
        self.data_path = data_path
        self.trader_data = None
        self.sentiment_data = None
        self.merged_data = None
        self.trader_profiles = None
        self.cluster_model = None
        self.performance_model = None
        
    def load_and_merge_data(self):
        """Load and merge trading data with sentiment data"""
        print("Loading datasets...")
        
        # Load trading data
        self.trader_data = pd.read_csv(f'{self.data_path}/raw/hyperliquid_trader_data.csv')
        print(f"Loaded {len(self.trader_data):,} trading records")
        
        # Load sentiment data
        self.sentiment_data = pd.read_csv(f'{self.data_path}/raw/bitcoin_sentiment_data.csv')
        print(f"Loaded {len(self.sentiment_data):,} sentiment records")
        
        # Clean and preprocess trading data
        self._clean_trading_data()
        
        # Clean and preprocess sentiment data
        self._clean_sentiment_data()
        
        # Merge datasets
        self._merge_datasets()
        
        print("Data loading and merging completed!")
        
    def _clean_trading_data(self):
        """Clean and preprocess trading data"""
        print("Cleaning trading data...")
        
        # Debug: print column names
        print(f"Available columns: {list(self.trader_data.columns)}")
        
        # Find timestamp column (could be 'Timestamp' or 'Timestamp IST')
        timestamp_col = None
        for col in self.trader_data.columns:
            if 'timestamp' in col.lower():
                timestamp_col = col
                break
        
        if timestamp_col is None:
            print("No timestamp column found, using first few rows as sample")
            print(self.trader_data.head())
            return
        
        print(f"Using timestamp column: {timestamp_col}")
        print(f"Sample timestamp values: {self.trader_data[timestamp_col].head()}")
        
        # Simple direct approach - just use pandas to_datetime with dayfirst=True
        self.trader_data['Timestamp'] = pd.to_datetime(self.trader_data[timestamp_col], dayfirst=True, errors='coerce')
        
        print(f"After timestamp conversion: {len(self.trader_data)} records")
        print(f"Sample converted timestamps: {self.trader_data['Timestamp'].head()}")
        print(f"Missing timestamps: {self.trader_data['Timestamp'].isna().sum()}")
        
        # Handle missing timestamps
        self.trader_data = self.trader_data.dropna(subset=['Timestamp'])
        
        # Convert numeric columns
        numeric_cols = ['Execution Price', 'Size Tokens', 'Size USD', 'Closed PnL']
        for col in numeric_cols:
            if col in self.trader_data.columns:
                self.trader_data[col] = pd.to_numeric(self.trader_data[col], errors='coerce')
        
        # Extract date for merging
        self.trader_data['date'] = self.trader_data['Timestamp'].dt.date
        
        # Clean account addresses
        self.trader_data['Account'] = self.trader_data['Account'].astype(str)
        
        print(f"Cleaned trading data: {len(self.trader_data):,} records")
        
    def _clean_sentiment_data(self):
        """Clean and preprocess sentiment data"""
        print("Cleaning sentiment data...")
        
        # Convert timestamp to datetime
        self.sentiment_data['date'] = pd.to_datetime(self.sentiment_data['date'])
        
        # Convert value to numeric
        self.sentiment_data['value'] = pd.to_numeric(self.sentiment_data['value'], errors='coerce')
        
        # Create binary sentiment classification
        self.sentiment_data['sentiment_binary'] = self.sentiment_data['classification'].map({
            'Extreme Fear': 0, 'Fear': 0, 'Neutral': 1, 'Greed': 2, 'Extreme Greed': 2
        })
        
        # Extract date for merging
        self.sentiment_data['date'] = self.sentiment_data['date'].dt.date
        
        print(f"Cleaned sentiment data: {len(self.sentiment_data):,} records")
        
    def _merge_datasets(self):
        """Merge trading and sentiment datasets"""
        print("Merging datasets...")
        
        # Merge on date
        self.merged_data = pd.merge(
            self.trader_data, 
            self.sentiment_data[['date', 'value', 'classification', 'sentiment_binary']], 
            on='date', 
            how='left'
        )
        
        # Forward fill sentiment for missing dates
        self.merged_data = self.merged_data.sort_values(['Account', 'Timestamp'])
        self.merged_data = self.merged_data.fillna(method='ffill')
        
        print(f"Merged dataset: {len(self.merged_data):,} records")
        
    def engineer_features(self):
        """Create comprehensive feature set for analysis"""
        print("Engineering features...")
        
        # Daily trader metrics
        daily_metrics = self.merged_data.groupby(['Account', 'date']).agg({
            'Closed PnL': ['sum', 'mean', 'std', 'count'],
            'Size USD': ['sum', 'mean'],
            'Execution Price': ['mean'],
            'value': ['mean', 'std'],  # sentiment metrics
            'sentiment_binary': ['mean']
        }).reset_index()
        
        # Flatten column names
        daily_metrics.columns = ['Account', 'date', 'daily_pnl', 'avg_pnl', 'pnl_std', 'trade_count', 
                               'daily_volume', 'avg_trade_size', 'avg_price', 'avg_sentiment', 
                               'sentiment_std', 'sentiment_binary']
        
        # Calculate additional metrics
        daily_metrics['win_rate'] = (daily_metrics['daily_pnl'] > 0).astype(int)
        daily_metrics['pnl_volatility'] = daily_metrics['pnl_std'].fillna(0)
        
        # Add lagged sentiment features
        daily_metrics = daily_metrics.sort_values(['Account', 'date'])
        daily_metrics['sentiment_lag1'] = daily_metrics.groupby('Account')['avg_sentiment'].shift(1)
        daily_metrics['sentiment_lag2'] = daily_metrics.groupby('Account')['avg_sentiment'].shift(2)
        
        # Add rolling metrics
        daily_metrics['pnl_ma7'] = daily_metrics.groupby('Account')['daily_pnl'].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
        daily_metrics['sentiment_ma7'] = daily_metrics.groupby('Account')['avg_sentiment'].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
        
        self.merged_data = daily_metrics
        print(f"Feature engineering completed: {len(self.merged_data):,} daily records")
        
    def create_trader_profiles(self):
        """Create comprehensive trader profiles"""
        print("Creating trader profiles...")
        
        # Aggregate trader-level statistics
        self.trader_profiles = self.merged_data.groupby('Account').agg({
            'daily_pnl': ['sum', 'mean', 'std', 'count'],
            'win_rate': 'mean',
            'daily_volume': 'sum',
            'avg_trade_size': 'mean',
            'avg_sentiment': 'mean',
            'pnl_volatility': 'mean',
            'sentiment_ma7': 'mean'
        }).reset_index()
        
        # Flatten column names
        self.trader_profiles.columns = ['Account', 'total_pnl', 'avg_daily_pnl', 'pnl_std', 'total_trades',
                                      'win_rate', 'total_volume', 'avg_trade_size', 'avg_sentiment',
                                      'avg_pnl_volatility', 'avg_sentiment_ma7']
        
        # Calculate additional metrics
        self.trader_profiles['sharpe_ratio'] = (self.trader_profiles['avg_daily_pnl'] / 
                                               self.trader_profiles['pnl_std']).fillna(0)
        self.trader_profiles['profit_factor'] = (self.trader_profiles['total_pnl'].clip(lower=0) / 
                                                abs(self.trader_profiles['total_pnl'].clip(upper=0))).fillna(0)
        
        # Remove outliers and infinite values
        self.trader_profiles = self.trader_profiles.replace([np.inf, -np.inf], np.nan)
        self.trader_profiles = self.trader_profiles.dropna()
        
        print(f"Created {len(self.trader_profiles):,} trader profiles")
        
    def analyze_sentiment_correlations(self):
        """Analyze correlations between sentiment and performance"""
        print("Analyzing sentiment correlations...")
        
        # Calculate correlations
        correlation_cols = ['daily_pnl', 'win_rate', 'daily_volume', 'pnl_volatility']
        sentiment_cols = ['avg_sentiment', 'sentiment_lag1', 'sentiment_lag2', 'sentiment_ma7']
        
        correlations = {}
        for perf_col in correlation_cols:
            if perf_col in self.merged_data.columns:
                corr_data = self.merged_data[[perf_col] + sentiment_cols].corr()
                correlations[perf_col] = corr_data[perf_col][sentiment_cols]
        
        # Performance comparison by sentiment periods
        sentiment_performance = self.merged_data.groupby('sentiment_binary').agg({
            'daily_pnl': ['mean', 'std', 'count'],
            'win_rate': 'mean',
            'daily_volume': 'mean'
        }).round(4)
        
        # Save results
        sentiment_performance.to_csv(f'{self.data_path}/results/sentiment_performance_analysis.csv')
        
        print("Sentiment correlation analysis completed!")
        return correlations, sentiment_performance
        
    def cluster_traders(self, n_clusters=4):
        """Cluster traders using K-means"""
        # Adjust number of clusters based on available data
        available_clusters = min(n_clusters, len(self.trader_profiles))
        print(f"Clustering traders into {available_clusters} groups (adjusted from {n_clusters})...")
        
        # Select features for clustering
        cluster_features = ['total_pnl', 'win_rate', 'total_trades', 'avg_trade_size', 
                          'avg_pnl_volatility', 'sharpe_ratio']
        
        # Prepare data for clustering
        cluster_data = self.trader_profiles[cluster_features].copy()
        
        # Standardize features
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
        
        # Apply K-means clustering
        self.cluster_model = KMeans(n_clusters=available_clusters, random_state=42, n_init=10)
        cluster_labels = self.cluster_model.fit_predict(cluster_data_scaled)
        
        # Add cluster labels to trader profiles
        self.trader_profiles['cluster'] = cluster_labels
        
        # Analyze cluster characteristics
        cluster_analysis = self.trader_profiles.groupby('cluster')[cluster_features].agg(['mean', 'std']).round(4)
        
        # Save cluster results
        self.trader_profiles.to_csv(f'{self.data_path}/results/trader_clusters.csv', index=False)
        cluster_analysis.to_csv(f'{self.data_path}/results/cluster_characteristics.csv')
        
        print("Trader clustering completed!")
        return cluster_analysis
        
    def build_performance_model(self):
        """Build predictive model for trader performance"""
        print("Building performance prediction model...")
        
        # Prepare features and target
        feature_cols = ['avg_sentiment', 'sentiment_lag1', 'sentiment_lag2', 'sentiment_ma7',
                       'daily_volume', 'avg_trade_size', 'pnl_ma7', 'pnl_volatility']
        
        # Remove rows with missing features
        model_data = self.merged_data[['daily_pnl'] + feature_cols].dropna()
        
        if len(model_data) < 100:
            print("Insufficient data for modeling")
            return None
            
        X = model_data[feature_cols]
        y = model_data['daily_pnl']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        self.performance_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.performance_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.performance_model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.performance_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save model and results
        with open(f'{self.data_path}/../models/trader_performance_model.pkl', 'wb') as f:
            pickle.dump(self.performance_model, f)
            
        feature_importance.to_csv(f'{self.data_path}/results/feature_importance.csv', index=False)
        
        print(f"Model performance - MSE: {mse:.4f}, R²: {r2:.4f}")
        return {'mse': mse, 'r2': r2, 'feature_importance': feature_importance}
        
    def calculate_risk_metrics(self):
        """Calculate comprehensive risk metrics"""
        print("Calculating risk metrics...")
        
        # Value at Risk calculations
        var_95 = self.merged_data['daily_pnl'].quantile(0.05)
        var_99 = self.merged_data['daily_pnl'].quantile(0.01)
        
        # Maximum drawdown
        try:
            cumulative_pnl = self.merged_data.groupby('Account')['daily_pnl'].cumsum()
            rolling_max = cumulative_pnl.groupby('Account').expanding().max()
            drawdown = (cumulative_pnl - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
        except KeyError:
            # If Account column doesn't exist, calculate overall drawdown
            cumulative_pnl = self.merged_data['daily_pnl'].cumsum()
            rolling_max = cumulative_pnl.expanding().max()
            drawdown = (cumulative_pnl - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
        
        # Risk metrics by sentiment
        risk_by_sentiment = self.merged_data.groupby('sentiment_binary').agg({
            'daily_pnl': ['std', 'skew'],
            'pnl_volatility': 'mean'
        }).round(4)
        
        # Save risk analysis
        risk_summary = pd.DataFrame({
            'metric': ['VaR_95', 'VaR_99', 'max_drawdown'],
            'value': [var_95, var_99, max_drawdown.min()]
        })
        
        risk_summary.to_csv(f'{self.data_path}/results/risk_metrics.csv', index=False)
        risk_by_sentiment.to_csv(f'{self.data_path}/results/risk_by_sentiment.csv')
        
        print("Risk metrics calculation completed!")
        return risk_summary, risk_by_sentiment
        
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("Generating visualizations...")
        
        # Set figure size and style
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Sentiment Time Series
        self._plot_sentiment_timeseries()
        
        # 2. Performance Comparison by Sentiment
        self._plot_performance_comparison()
        
        # 3. Correlation Heatmap
        self._plot_correlation_heatmap()
        
        # 4. Trader Clusters
        self._plot_trader_clusters()
        
        # 5. Feature Importance
        self._plot_feature_importance()
        
        # 6. Risk Analysis
        self._plot_risk_analysis()
        
        print("All visualizations generated!")
        
    def _plot_sentiment_timeseries(self):
        """Plot Fear & Greed Index over time"""
        plt.figure(figsize=(15, 8))
        
        # Get unique dates and sentiment values
        sentiment_plot = self.sentiment_data.groupby('date')['value'].mean().reset_index()
        sentiment_plot['date'] = pd.to_datetime(sentiment_plot['date'])
        
        plt.plot(sentiment_plot['date'], sentiment_plot['value'], linewidth=2, color='#2E86AB')
        plt.fill_between(sentiment_plot['date'], sentiment_plot['value'], alpha=0.3, color='#2E86AB')
        
        # Add sentiment zones
        plt.axhline(y=25, color='red', linestyle='--', alpha=0.7, label='Extreme Fear')
        plt.axhline(y=45, color='orange', linestyle='--', alpha=0.7, label='Fear')
        plt.axhline(y=55, color='yellow', linestyle='--', alpha=0.7, label='Neutral')
        plt.axhline(y=75, color='lightgreen', linestyle='--', alpha=0.7, label='Greed')
        plt.axhline(y=75, color='green', linestyle='--', alpha=0.7, label='Extreme Greed')
        
        plt.title('Bitcoin Fear & Greed Index Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Fear & Greed Index', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(f'{self.data_path}/../visualizations/sentiment_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_performance_comparison(self):
        """Plot performance comparison between sentiment periods"""
        plt.figure(figsize=(12, 8))
        
        # Performance by sentiment
        perf_by_sentiment = self.merged_data.groupby('sentiment_binary').agg({
            'daily_pnl': 'mean',
            'win_rate': 'mean'
        }).reset_index()
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PnL comparison
        sentiment_labels = ['Fear', 'Neutral', 'Greed']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        bars1 = ax1.bar(sentiment_labels, perf_by_sentiment['daily_pnl'], color=colors, alpha=0.8)
        ax1.set_title('Average Daily PnL by Sentiment', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Daily PnL (USD)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}', ha='center', va='bottom')
        
        # Win rate comparison
        bars2 = ax2.bar(sentiment_labels, perf_by_sentiment['win_rate'] * 100, color=colors, alpha=0.8)
        ax2.set_title('Win Rate by Sentiment', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Win Rate (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.data_path}/../visualizations/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_correlation_heatmap(self):
        """Plot correlation heatmap between sentiment and performance"""
        plt.figure(figsize=(10, 8))
        
        # Select correlation columns
        corr_cols = ['daily_pnl', 'win_rate', 'daily_volume', 'pnl_volatility', 
                     'avg_sentiment', 'sentiment_lag1', 'sentiment_lag2']
        
        corr_data = self.merged_data[corr_cols].corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Correlation Heatmap: Sentiment vs Performance', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(f'{self.data_path}/../visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_trader_clusters(self):
        """Plot trader clusters"""
        if self.trader_profiles is None or 'cluster' not in self.trader_profiles.columns:
            print("No cluster data available")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Scatter plot of clusters
        scatter = plt.scatter(self.trader_profiles['total_pnl'], 
                            self.trader_profiles['win_rate'],
                            c=self.trader_profiles['cluster'], 
                            cmap='viridis', alpha=0.7, s=50)
        
        plt.xlabel('Total PnL (USD)', fontsize=12)
        plt.ylabel('Win Rate', fontsize=12)
        plt.title('Trader Clusters: PnL vs Win Rate', fontsize=16, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.data_path}/../visualizations/cluster_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_feature_importance(self):
        """Plot feature importance from predictive model"""
        if self.performance_model is None:
            print("No model available for feature importance")
            return
            
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': ['avg_sentiment', 'sentiment_lag1', 'sentiment_lag2', 'sentiment_ma7',
                       'daily_volume', 'avg_trade_size', 'pnl_ma7', 'pnl_volatility'],
            'importance': self.performance_model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        
        bars = plt.barh(range(len(feature_importance)), feature_importance['importance'], 
                       color='#2E86AB', alpha=0.8)
        
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title('Feature Importance in Performance Prediction', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(f'{self.data_path}/../visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_risk_analysis(self):
        """Plot risk analysis charts"""
        plt.figure(figsize=(15, 6))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PnL distribution by sentiment
        sentiment_labels = ['Fear', 'Neutral', 'Greed']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, sentiment in enumerate([0, 1, 2]):
            sentiment_data = self.merged_data[self.merged_data['sentiment_binary'] == sentiment]['daily_pnl']
            ax1.hist(sentiment_data, bins=30, alpha=0.7, color=colors[i], 
                    label=sentiment_labels[i], density=True)
        
        ax1.set_xlabel('Daily PnL (USD)', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('PnL Distribution by Sentiment', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volatility comparison
        vol_by_sentiment = self.merged_data.groupby('sentiment_binary')['pnl_volatility'].mean()
        bars = ax2.bar(sentiment_labels, vol_by_sentiment, color=colors, alpha=0.8)
        ax2.set_title('Average PnL Volatility by Sentiment', fontsize=14, fontweight='bold')
        ax2.set_ylabel('PnL Volatility', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.data_path}/../visualizations/risk_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting comprehensive cryptocurrency trading analysis...")
        print("=" * 60)
        
        # Step 1: Load and merge data
        self.load_and_merge_data()
        
        # Step 2: Feature engineering
        self.engineer_features()
        
        # Step 3: Create trader profiles
        self.create_trader_profiles()
        
        # Step 4: Sentiment correlation analysis
        correlations, sentiment_performance = self.analyze_sentiment_correlations()
        
        # Step 5: Trader clustering
        cluster_analysis = self.cluster_traders()
        
        # Step 6: Predictive modeling
        model_results = self.build_performance_model()
        
        # Step 7: Risk analysis
        risk_summary, risk_by_sentiment = self.calculate_risk_metrics()
        
        # Step 8: Generate visualizations
        self.generate_visualizations()
        
        print("\nAnalysis completed successfully!")
        print("=" * 60)
        
        # Print key findings
        self._print_key_findings()
        
        return {
            'correlations': correlations,
            'sentiment_performance': sentiment_performance,
            'cluster_analysis': cluster_analysis,
            'model_results': model_results,
            'risk_summary': risk_summary,
            'risk_by_sentiment': risk_by_sentiment
        }
        
    def _print_key_findings(self):
        """Print key findings from the analysis"""
        print("\nKEY FINDINGS:")
        print("-" * 40)
        
        # Sentiment performance
        if hasattr(self, 'merged_data') and len(self.merged_data) > 0:
            fear_perf = self.merged_data[self.merged_data['sentiment_binary'] == 0]['daily_pnl'].mean()
            greed_perf = self.merged_data[self.merged_data['sentiment_binary'] == 2]['daily_pnl'].mean()
            print(f"• Fear periods average PnL: ${fear_perf:.2f}")
            print(f"• Greed periods average PnL: ${greed_perf:.2f}")
            print(f"• Performance difference: ${greed_perf - fear_perf:.2f}")
        
        # Trader clustering
        if hasattr(self, 'trader_profiles') and 'cluster' in self.trader_profiles.columns:
            n_clusters = self.trader_profiles['cluster'].nunique()
            print(f"• Identified {n_clusters} distinct trader archetypes")
        
        # Model performance
        if hasattr(self, 'performance_model'):
            print("• Predictive model successfully built for trader performance")
        
        print("\nAll results saved to data/results/ directory")
        print("All visualizations saved to visualizations/ directory")

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = CryptoTradingAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    print("\nAnalysis completed! Check the results and visualizations directories.") 