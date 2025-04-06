"""
Azerbaijan Trend Detection System
================================
A comprehensive trend monitoring system that tracks economic, social, political,
and cultural indicators within Azerbaijan.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import tweepy
import warnings
warnings.filterwarnings('ignore')

class AzerbaijanTrendDetector:
    """Main class to detect and analyze trends across multiple domains in Azerbaijan."""
    
    def __init__(self, api_keys=None):
        """
        Initialize the trend detector with necessary API keys.
        
        Parameters:
        api_keys (dict): Dictionary containing API keys for various data sources.
        """
        self.api_keys = api_keys or {}
        self.economic_data = None
        self.social_data = None
        self.political_data = None
        self.media_data = None
        
        # Download necessary NLTK resources
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            # Add Azerbaijani stop words
            self.az_stop_words = set(['və', 'bu', 'ilə', 'bir', 'üçün', 'olar', 'olur', 'daha'])
            self.stop_words.update(self.az_stop_words)
        except:
            print("Warning: NLTK resources could not be downloaded. Text analysis may be limited.")
            self.stop_words = set()
    
    def fetch_economic_indicators(self, start_date='2020-01-01', end_date=None):
        """
        Fetch economic indicators for Azerbaijan.
        
        Parameters:
        start_date (str): Start date for data collection in YYYY-MM-DD format.
        end_date (str): End date for data collection in YYYY-MM-DD format.
        
        Returns:
        pandas.DataFrame: DataFrame containing economic indicators.
        """
        print("Fetching economic indicators...")
        
        # Placeholder for actual API calls to sources like:
        # - Central Bank of Azerbaijan (https://www.cbar.az/)
        # - State Statistical Committee (https://www.stat.gov.az/)
        # - World Bank API
        # - IMF data
        
        # Sample data structure (would be populated with real API data)
        date_range = pd.date_range(start=start_date, end=end_date or pd.Timestamp.today(), freq='M')
        
        # Generate sample data for demonstration
        np.random.seed(42)  # For reproducibility
        gdp_growth = np.cumsum(np.random.normal(0.5, 1, size=len(date_range))) + 2
        inflation = np.abs(np.cumsum(np.random.normal(0, 0.3, size=len(date_range)))) + 3
        oil_prices = np.cumsum(np.random.normal(0, 2, size=len(date_range))) + 70
        non_oil_gdp = np.cumsum(np.random.normal(0.3, 0.5, size=len(date_range))) + 1.5
        exchange_rate = np.random.normal(1.7, 0.02, size=len(date_range))
        
        self.economic_data = pd.DataFrame({
            'date': date_range,
            'gdp_growth': gdp_growth,
            'inflation': inflation,
            'oil_prices': oil_prices,
            'non_oil_gdp': non_oil_gdp,
            'exchange_rate': exchange_rate
        })
        
        return self.economic_data
    
    def fetch_social_indicators(self, start_date='2020-01-01', end_date=None):
        """
        Fetch social indicators for Azerbaijan.
        
        Parameters:
        start_date (str): Start date for data collection in YYYY-MM-DD format.
        end_date (str): End date for data collection in YYYY-MM-DD format.
        
        Returns:
        pandas.DataFrame: DataFrame containing social indicators.
        """
        print("Fetching social indicators...")
        
        # Placeholder for actual API calls to sources like:
        # - Azerbaijan State Statistical Committee
        # - Ministry of Labour and Social Protection
        # - World Bank Social Indicators
        # - UN Development Programme

        # Sample data structure
        date_range = pd.date_range(start=start_date, end=end_date or pd.Timestamp.today(), freq='M')
        
        # Generate sample data
        np.random.seed(43)
        unemployment = np.clip(np.cumsum(np.random.normal(0, 0.2, size=len(date_range))) + 5, 3, 8)
        literacy_rate = np.clip(np.random.normal(99.8, 0.1, size=len(date_range)), 99.5, 100)
        internet_penetration = np.clip(np.cumsum(np.random.normal(0.3, 0.1, size=len(date_range))) + 80, 80, 95)
        urban_population = np.clip(np.cumsum(np.random.normal(0.1, 0.05, size=len(date_range))) + 56, 56, 60)
        
        self.social_data = pd.DataFrame({
            'date': date_range,
            'unemployment': unemployment,
            'literacy_rate': literacy_rate,
            'internet_penetration': internet_penetration,
            'urban_population': urban_population
        })
        
        return self.social_data
    
    def fetch_political_indicators(self, start_date='2020-01-01', end_date=None):
        """
        Fetch political indicators and government activities.
        
        Parameters:
        start_date (str): Start date for data collection in YYYY-MM-DD format.
        end_date (str): End date for data collection in YYYY-MM-DD format.
        
        Returns:
        pandas.DataFrame: DataFrame containing political indicators.
        """
        print("Fetching political indicators...")
        
        # Placeholder for actual data from sources like:
        # - President's Office website (https://president.az/en)
        # - Cabinet of Ministers
        # - Ministry of Foreign Affairs
        # - Parliament of Azerbaijan (Milli Majlis)
        
        # Sample data structure
        date_range = pd.date_range(start=start_date, end=end_date or pd.Timestamp.today(), freq='M')
        
        # Generate sample data
        np.random.seed(44)
        legislation_count = np.random.randint(5, 20, size=len(date_range))
        international_agreements = np.random.randint(0, 5, size=len(date_range))
        cabinet_changes = np.random.randint(0, 2, size=len(date_range))
        public_appearances = np.random.randint(5, 15, size=len(date_range))
        
        self.political_data = pd.DataFrame({
            'date': date_range,
            'legislation_count': legislation_count,
            'international_agreements': international_agreements,
            'cabinet_changes': cabinet_changes,
            'public_appearances': public_appearances
        })
        
        return self.political_data
    
    def scrape_news_sources(self, num_articles=100):
        """
        Scrape news articles from Azerbaijani news sources.
        
        Parameters:
        num_articles (int): Number of articles to scrape.
        
        Returns:
        pandas.DataFrame: DataFrame containing news articles and metadata.
        """
        print(f"Scraping latest news articles from Azerbaijani sources (limit: {num_articles})...")
        
        # Placeholder for actual web scraping from sources like:
        # - AzerNews (https://www.azernews.az/)
        # - Trend News Agency (https://en.trend.az/)
        # - APA News (https://apa.az/en/)
        # - Report News Agency (https://report.az/en/)
        
        # Sample data for demonstration
        sources = ['AzerNews', 'Trend', 'APA', 'Report', 'Azertag']
        categories = ['Politics', 'Economy', 'Society', 'Culture', 'International']
        
        np.random.seed(45)
        
        # Generate sample news data
        news_data = []
        
        sample_headlines = [
            "Azerbaijan's GDP grows by {0}% in Q{1}",
            "New infrastructure project launched in {0}",
            "Azerbaijan signs agreement with {0} on economic cooperation",
            "Tourism in Azerbaijan increases by {0}% compared to last year",
            "New cultural center opened in {0}",
            "Azerbaijan's foreign trade turnover increases by {0}%",
            "President meets with delegation from {0}",
            "Azerbaijan participates in international forum on {0}",
            "New technology park to be built in {0}",
            "Azerbaijan's non-oil sector shows {0}% growth"
        ]
        
        cities = ['Baku', 'Ganja', 'Sumgayit', 'Mingachevir', 'Shirvan', 'Nakhchivan']
        countries = ['Turkey', 'Russia', 'Georgia', 'Iran', 'EU', 'China', 'USA', 'UK']
        topics = ['renewable energy', 'digital transformation', 'agriculture', 'education', 'healthcare']
        
        for i in range(num_articles):
            date = pd.Timestamp.today() - pd.Timedelta(days=np.random.randint(0, 30))
            source = np.random.choice(sources)
            category = np.random.choice(categories)
            
            # Generate headline
            headline_template = np.random.choice(sample_headlines)
            if '{0}' in headline_template and '{1}' in headline_template:
                headline = headline_template.format(np.random.randint(3, 8), np.random.randint(1, 4))
            elif '{0}' in headline_template:
                if 'growth' in headline_template or 'increases' in headline_template:
                    headline = headline_template.format(np.random.randint(3, 15))
                elif 'delegation' in headline_template or 'agreement' in headline_template:
                    headline = headline_template.format(np.random.choice(countries))
                elif 'project' in headline_template or 'center' in headline_template or 'park' in headline_template:
                    headline = headline_template.format(np.random.choice(cities))
                elif 'forum' in headline_template:
                    headline = headline_template.format(np.random.choice(topics))
                else:
                    headline = headline_template.format(np.random.choice(cities))
            else:
                headline = headline_template
            
            # Generate simple content
            content = f"This is a sample article about {headline.lower()}. " \
                      f"The development was announced by officials from the {np.random.choice(['Ministry of Economy', 'Ministry of Finance', 'Ministry of Foreign Affairs', 'Cabinet of Ministers'])}. " \
                      f"Experts believe this will have a positive impact on the {np.random.choice(['economy', 'society', 'international relations', 'region'])}"
            
            views = np.random.randint(100, 5000)
            shares = np.random.randint(5, 500)
            
            news_data.append({
                'date': date,
                'source': source,
                'category': category,
                'headline': headline,
                'content': content,
                'views': views,
                'shares': shares
            })
        
        self.media_data = pd.DataFrame(news_data)
        return self.media_data
    
    def analyze_social_media_trends(self, platform='twitter', keywords=None, limit=100):
        """
        Analyze social media trends related to Azerbaijan.
        
        Parameters:
        platform (str): Social media platform to analyze ('twitter', 'instagram', etc.)
        keywords (list): List of keywords to search for.
        limit (int): Maximum number of posts to analyze.
        
        Returns:
        dict: Dictionary containing trend analysis results.
        """
        print(f"Analyzing {platform} trends for Azerbaijan-related content...")
        
        if keywords is None:
            keywords = ['Azerbaijan', 'Baku', 'Azerbaijani', 'Azərbaycan']
            
        # Placeholder for actual API calls to social media platforms
        # In a real implementation, you would use platform-specific APIs:
        # - Twitter API (via tweepy)
        # - Instagram API
        # - Facebook Graph API
        # - TikTok API
        
        # Sample return structure
        trending_topics = [
            {'topic': 'Economy', 'count': 342, 'sentiment': 0.65},
            {'topic': 'Tourism', 'count': 289, 'sentiment': 0.81},
            {'topic': 'Culture', 'count': 156, 'sentiment': 0.72},
            {'topic': 'Tech', 'count': 124, 'sentiment': 0.69},
            {'topic': 'Sports', 'count': 98, 'sentiment': 0.54}
        ]
        
        top_hashtags = [
            {'hashtag': '#Azerbaijan', 'count': 567},
            {'hashtag': '#Baku', 'count': 423},
            {'hashtag': '#CaspianSea', 'count': 201},
            {'hashtag': '#AzerbaijaniCuisine', 'count': 187},
            {'hashtag': '#AzerbaijanGP', 'count': 156}
        ]
        
        influential_accounts = [
            {'account': 'Official_Azerbaijan', 'followers': 124500, 'engagement': 0.87},
            {'account': 'VisitAzerbaijan', 'followers': 98700, 'engagement': 0.92},
            {'account': 'AzerbaijanMFA', 'followers': 87600, 'engagement': 0.75},
            {'account': 'BakuTourism', 'followers': 76500, 'engagement': 0.83},
            {'account': 'AzerbaijanBusiness', 'followers': 65400, 'engagement': 0.71}
        ]
        
        return {
            'platform': platform,
            'keywords': keywords,
            'sample_size': limit,
            'trending_topics': trending_topics,
            'top_hashtags': top_hashtags,
            'influential_accounts': influential_accounts
        }
    
    def detect_economic_trends(self, window=12):
        """
        Detect trends in economic indicators.
        
        Parameters:
        window (int): Rolling window size for trend detection in months.
        
        Returns:
        dict: Dictionary containing detected economic trends.
        """
        if self.economic_data is None:
            raise ValueError("Economic data not loaded. Run fetch_economic_indicators() first.")
        
        print("Detecting economic trends...")
        
        # Calculate rolling averages to identify trends
        econ_trends = self.economic_data.copy()
        for col in econ_trends.columns:
            if col != 'date':
                econ_trends[f'{col}_rolling'] = econ_trends[col].rolling(window=window, min_periods=1).mean()
                econ_trends[f'{col}_trend'] = np.where(
                    econ_trends[col] > econ_trends[f'{col}_rolling'], 
                    'Rising', 
                    np.where(econ_trends[col] < econ_trends[f'{col}_rolling'], 'Falling', 'Stable')
                )
        
        # Get the most recent trends
        latest_trends = econ_trends.iloc[-1]
        
        # Calculate YoY changes
        if len(econ_trends) >= 12:
            for col in [c for c in econ_trends.columns if c != 'date' and not c.endswith('_rolling') and not c.endswith('_trend')]:
                econ_trends[f'{col}_yoy'] = econ_trends[col].pct_change(periods=12) * 100
        
        return {
            'latest_date': latest_trends['date'].strftime('%Y-%m-%d'),
            'gdp_growth': {
                'value': latest_trends['gdp_growth'],
                'trend': latest_trends['gdp_growth_trend'],
                'yoy_change': latest_trends.get('gdp_growth_yoy', 'N/A')
            },
            'inflation': {
                'value': latest_trends['inflation'],
                'trend': latest_trends['inflation_trend'],
                'yoy_change': latest_trends.get('inflation_yoy', 'N/A')
            },
            'oil_prices': {
                'value': latest_trends['oil_prices'],
                'trend': latest_trends['oil_prices_trend'],
                'yoy_change': latest_trends.get('oil_prices_yoy', 'N/A')
            },
            'non_oil_gdp': {
                'value': latest_trends['non_oil_gdp'],
                'trend': latest_trends['non_oil_gdp_trend'],
                'yoy_change': latest_trends.get('non_oil_gdp_yoy', 'N/A')
            },
            'exchange_rate': {
                'value': latest_trends['exchange_rate'],
                'trend': latest_trends['exchange_rate_trend'],
                'yoy_change': latest_trends.get('exchange_rate_yoy', 'N/A')
            }
        }
    
    def detect_social_trends(self, window=12):
        """
        Detect trends in social indicators.
        
        Parameters:
        window (int): Rolling window size for trend detection in months.
        
        Returns:
        dict: Dictionary containing detected social trends.
        """
        if self.social_data is None:
            raise ValueError("Social data not loaded. Run fetch_social_indicators() first.")
        
        print("Detecting social trends...")
        
        # Similar approach to economic trends
        social_trends = self.social_data.copy()
        for col in social_trends.columns:
            if col != 'date':
                social_trends[f'{col}_rolling'] = social_trends[col].rolling(window=window, min_periods=1).mean()
                social_trends[f'{col}_trend'] = np.where(
                    social_trends[col] > social_trends[f'{col}_rolling'], 
                    'Rising', 
                    np.where(social_trends[col] < social_trends[f'{col}_rolling'], 'Falling', 'Stable')
                )
        
        # Get the most recent trends
        latest_trends = social_trends.iloc[-1]
        
        # Calculate YoY changes
        if len(social_trends) >= 12:
            for col in [c for c in social_trends.columns if c != 'date' and not c.endswith('_rolling') and not c.endswith('_trend')]:
                social_trends[f'{col}_yoy'] = social_trends[col].pct_change(periods=12) * 100
        
        return {
            'latest_date': latest_trends['date'].strftime('%Y-%m-%d'),
            'unemployment': {
                'value': latest_trends['unemployment'],
                'trend': latest_trends['unemployment_trend'],
                'yoy_change': latest_trends.get('unemployment_yoy', 'N/A')
            },
            'literacy_rate': {
                'value': latest_trends['literacy_rate'],
                'trend': latest_trends['literacy_rate_trend'],
                'yoy_change': latest_trends.get('literacy_rate_yoy', 'N/A')
            },
            'internet_penetration': {
                'value': latest_trends['internet_penetration'],
                'trend': latest_trends['internet_penetration_trend'],
                'yoy_change': latest_trends.get('internet_penetration_yoy', 'N/A')
            },
            'urban_population': {
                'value': latest_trends['urban_population'],
                'trend': latest_trends['urban_population_trend'],
                'yoy_change': latest_trends.get('urban_population_yoy', 'N/A')
            }
        }
    
    def detect_political_trends(self, window=6):
        """
        Detect trends in political indicators.
        
        Parameters:
        window (int): Rolling window size for trend detection in months.
        
        Returns:
        dict: Dictionary containing detected political trends.
        """
        if self.political_data is None:
            raise ValueError("Political data not loaded. Run fetch_political_indicators() first.")
        
        print("Detecting political trends...")
        
        # Calculate averages and identify outliers/anomalies
        pol_trends = self.political_data.copy()
        for col in pol_trends.columns:
            if col != 'date':
                # Calculate rolling statistics
                pol_trends[f'{col}_rolling'] = pol_trends[col].rolling(window=window, min_periods=1).mean()
                pol_trends[f'{col}_std'] = pol_trends[col].rolling(window=window, min_periods=1).std()
                
                # Identify significant changes
                pol_trends[f'{col}_zscore'] = (pol_trends[col] - pol_trends[f'{col}_rolling']) / pol_trends[f'{col}_std'].replace(0, 1)
                pol_trends[f'{col}_significant'] = abs(pol_trends[f'{col}_zscore']) > 1.96  # 95% confidence
        
        # Get the most recent data
        latest_data = pol_trends.iloc[-1]
        
        # Identify trends in last 3 months compared to previous periods
        recent_period = pol_trends.iloc[-3:].mean()
        previous_period = pol_trends.iloc[-6:-3].mean() if len(pol_trends) >= 6 else None
        
        trends = {}
        for col in ['legislation_count', 'international_agreements', 'cabinet_changes', 'public_appearances']:
            if previous_period is not None:
                if recent_period[col] > previous_period[col] * 1.2:
                    trend = 'Significant increase'
                elif recent_period[col] > previous_period[col] * 1.05:
                    trend = 'Slight increase'
                elif recent_period[col] < previous_period[col] * 0.8:
                    trend = 'Significant decrease'
                elif recent_period[col] < previous_period[col] * 0.95:
                    trend = 'Slight decrease'
                else:
                    trend = 'Stable'
            else:
                trend = 'Insufficient data'
            
            trends[col] = {
                'recent_value': latest_data[col],
                'trend': trend,
                'significant_change': latest_data[f'{col}_significant']
            }
        
        return trends
    
    def analyze_news_topics(self, min_clusters=5, max_clusters=10):
        """
        Analyze trending topics in news articles.
        
        Parameters:
        min_clusters (int): Minimum number of topic clusters to identify.
        max_clusters (int): Maximum number of topic clusters to identify.
        
        Returns:
        dict: Dictionary containing news topic analysis.
        """
        if self.media_data is None:
            raise ValueError("News data not loaded. Run scrape_news_sources() first.")
        
        print("Analyzing news topics...")
        
        # Text preprocessing
        def preprocess_text(text):
            text = text.lower()
            # Remove URLs, special characters, etc.
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            # Tokenize
            tokens = word_tokenize(text)
            # Remove stopwords
            filtered_tokens = [w for w in tokens if w not in self.stop_words]
            return ' '.join(filtered_tokens)
        
        # Prepare corpus
        corpus = self.media_data['content'].apply(preprocess_text).tolist()
        
        # Extract features using TF-IDF
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(corpus)
        
        # Determine optimal number of clusters (simplified approach)
        optimal_k = min_clusters
        
        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        self.media_data['cluster'] = kmeans.fit_predict(X)
        
        # Extract top terms per cluster
        feature_names = vectorizer.get_feature_names_out()
        centroids = kmeans.cluster_centers_
        
        topics = []
        for i in range(optimal_k):
            # Get top terms for this cluster
            indices = centroids[i].argsort()[-10:][::-1]  # Top 10 terms
            top_terms = [feature_names[j] for j in indices]
            
            # Get articles in this cluster
            cluster_articles = self.media_data[self.media_data['cluster'] == i]
            
            topics.append({
                'id': i,
                'size': len(cluster_articles),
                'percentage': len(cluster_articles) / len(self.media_data) * 100,
                'top_terms': top_terms,
                'top_headlines': cluster_articles['headline'].tolist()[:5],
                'sources': cluster_articles['source'].value_counts().to_dict(),
                'avg_engagement': (cluster_articles['views'].mean() + cluster_articles['shares'].mean()) / 2
            })
        
        # Sort topics by size
        topics = sorted(topics, key=lambda x: x['size'], reverse=True)
        
        return {
            'total_articles': len(self.media_data),
            'analysis_date': pd.Timestamp.today().strftime('%Y-%m-%d'),
            'optimal_clusters': optimal_k,
            'topics': topics
        }
    
    def forecast_indicators(self, indicators=None, periods=12):
        """
        Forecast future values for key indicators.
        
        Parameters:
        indicators (list): List of indicators to forecast.
        periods (int): Number of periods (months) to forecast.
        
        Returns:
        dict: Dictionary containing forecasts for specified indicators.
        """
        print(f"Forecasting indicators for the next {periods} months...")
        
        if indicators is None:
            indicators = ['gdp_growth', 'inflation', 'oil_prices', 'exchange_rate', 'unemployment']
        
        forecasts = {}
        
        # Economic indicators
        if self.economic_data is not None:
            for indicator in [i for i in indicators if i in self.economic_data.columns and i != 'date']:
                # Prepare data for Prophet
                df = pd.DataFrame({
                    'ds': self.economic_data['date'],
                    'y': self.economic_data[indicator]
                })
                
                # Train model
                model = Prophet(interval_width=0.95)
                model.fit(df)
                
                # Make future dataframe and predict
                future = model.make_future_dataframe(periods=periods, freq='M')
                forecast = model.predict(future)
                
                # Extract relevant parts
                historical = forecast[forecast['ds'].isin(df['ds'])][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                predicted = forecast[~forecast['ds'].isin(df['ds'])][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                
                forecasts[indicator] = {
                    'historical': historical.rename(columns={
                        'ds': 'date', 
                        'yhat': 'value', 
                        'yhat_lower': 'lower_bound', 
                        'yhat_upper': 'upper_bound'
                    }).to_dict('records'),
                    'forecast': predicted.rename(columns={
                        'ds': 'date', 
                        'yhat': 'value', 
                        'yhat_lower': 'lower_bound', 
                        'yhat_upper': 'upper_bound'
                    }).to_dict('records')
                }
        
        # Social indicators
        if self.social_data is not None:
            for indicator in [i for i in indicators if i in self.social_data.columns and i != 'date']:
                # Use same approach as economic indicators
                df = pd.DataFrame({
                    'ds': self.social_data['date'],
                    'y': self.social_data[indicator]
                })
                
                model = Prophet(interval_width=0.95)
                model.fit(df)
                
                future = model.make_future_dataframe(periods=periods, freq='M')
                forecast = model.predict(future)
                
                historical = forecast[forecast['ds'].isin(df['ds'])][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                predicted = forecast[~forecast['ds'].isin(df['ds'])][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                
                forecasts[indicator] = {
                    'historical': historical.rename(columns={
                        'ds': 'date', 
                        'yhat': 'value', 
                        'yhat_lower': 'lower_bound', 
                        'yhat_upper': 'upper_bound'
                    }).to_dict('records'),
                    'forecast': predicted.rename(columns={
                        'ds': 'date', 
                        'yhat': 'value', 
                        'yhat_lower': 'lower_bound', 
                        'yhat_upper': 'upper_bound'
                    }).to_dict('records')
                }
        
        return forecasts
    
    def generate_dashboard_data(self):
        """
        Generate a comprehensive data package for a dashboard.
        
        Returns:
        dict: Dictionary containing all dashboard data.
        """
        print("Generating dashboard data...")
        
        dashboard = {
            'generated_at': pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S'),
            'country': 'Azerbaijan',
            'timeframe': f"{self.economic_data.iloc[0]['date'].strftime('%Y-%m-%d')} to {self.economic_data.iloc[-1]['date'].strftime('%Y-%m-%d')}" if self.economic_data is not None else 'N/A'
        }
        
        # Add trend data if available
        try:
            dashboard['economic_trends'] = self.detect_economic_trends()
        except:
            dashboard['economic_trends'] = "Data not available"
            
        try:
            dashboard['social_trends'] = self.detect_social_trends()
        except:
            dashboard['social_trends'] = "Data not available"
            
        try:
            dashboard['political_trends'] = self.detect_political_trends()
        except:
            dashboard['political_trends'] = "Data not available"
            
        try:
            dashboard['news_analysis'] = self.analyze_news_topics()
        except:
            dashboard['news_analysis'] = "Data not available"