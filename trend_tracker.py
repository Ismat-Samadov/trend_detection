import pandas as pd
from pytrends.request import TrendReq
import time
import datetime
import os
import logging
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trend_tracker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

class SimpleTrendTracker:
    def __init__(self):
        """Initialize the tracker with basic configuration"""
        # Use more robust connection settings
        self.pytrends = TrendReq(
            hl='en-US', 
            tz=360,
            timeout=(10, 25),
            retries=2,
            backoff_factor=1
        )
        self.output_dir = "trend_data"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
    
    def _save_to_csv(self, df, prefix):
        """Helper method to save a dataframe to CSV with timestamp"""
        if df is None or df.empty:
            logger.warning(f"No data to save for {prefix}")
            return None
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        df.to_csv(filepath)
        logger.info(f"Saved {prefix} data to {filepath}")
        return filepath
    
    def get_daily_trends(self):
        """Get daily search trends"""
        try:
            logger.info("Fetching daily search trends...")
            # Try to get trending searches - use US as fallback if needed
            try:
                df = self.pytrends.trending_searches(pn='united_states')
            except Exception as e:
                logger.warning(f"Error with default trending_searches: {e}")
                # Try another country
                countries = ['united_kingdom', 'india', 'canada', 'australia']
                random_country = random.choice(countries)
                logger.info(f"Trying with country: {random_country}")
                df = self.pytrends.trending_searches(pn=random_country)
            
            # Save the results
            self._save_to_csv(df, "daily_trends")
            return df
            
        except Exception as e:
            logger.error(f"Error getting daily trends: {e}")
            return None
    
    def track_keywords(self, keywords, timeframe='now 7-d', geo=''):
        """
        Track specific keywords and save results
        
        Args:
            keywords: List of keywords to track (max 5)
            timeframe: Time period to analyze
            geo: Geographic location (empty for worldwide)
        """
        if not keywords:
            logger.warning("No keywords provided")
            return None
            
        # Limit to 5 keywords (Google Trends limitation)
        keywords = keywords[:5]
        logger.info(f"Tracking keywords: {keywords}")
        
        try:
            # Build the payload
            self.pytrends.build_payload(
                keywords, 
                cat=0, 
                timeframe=timeframe, 
                geo=geo
            )
            
            # Get interest over time
            time.sleep(1)  # Brief pause to avoid rate limits
            interest_df = self.pytrends.interest_over_time()
            self._save_to_csv(interest_df, f"keyword_interest")
            
            # Get related queries
            time.sleep(1)  # Brief pause to avoid rate limits
            related_queries = self.pytrends.related_queries()
            
            # Save related queries for each keyword
            for keyword in keywords:
                if keyword in related_queries and related_queries[keyword]['top'] is not None:
                    self._save_to_csv(related_queries[keyword]['top'], f"related_{keyword}")
            
            return {
                'interest': interest_df,
                'related': related_queries
            }
            
        except Exception as e:
            logger.error(f"Error tracking keywords: {e}")
            return None
    
    def get_realtime_trends(self):
        """Get real-time trending searches"""
        try:
            logger.info("Fetching real-time trending searches...")
            df = self.pytrends.realtime_trending_searches(pn='US')
            self._save_to_csv(df, "realtime_trends")
            return df
        except Exception as e:
            logger.error(f"Error getting real-time trends: {e}")
            return None
    
    def run_basic_monitoring(self, custom_keywords=None, interval_seconds=3600, max_iterations=None):
        """
        Run basic monitoring cycle
        
        Args:
            custom_keywords: Optional list of keywords to track
            interval_seconds: How often to check (default: hourly)
            max_iterations: Maximum number of iterations (None for unlimited)
        """
        # Default keywords if none provided
        default_keywords = ['AI', 'machine learning', 'python', 'data science', 'technology']
        keywords_to_track = custom_keywords if custom_keywords else default_keywords
        
        logger.info(f"Starting basic monitoring with {interval_seconds}s interval")
        logger.info(f"Will track keywords: {keywords_to_track}")
        
        iteration = 1
        try:
            while True:
                logger.info(f"Starting iteration {iteration}")
                
                # Get trending topics (try both methods)
                try:
                    self.get_daily_trends()
                except Exception as e:
                    logger.error(f"Failed to get daily trends: {e}")
                
                try:
                    self.get_realtime_trends()
                except Exception as e:
                    logger.error(f"Failed to get realtime trends: {e}")
                
                # Track specified keywords
                self.track_keywords(keywords_to_track)
                
                # Check if we've reached maximum iterations
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Reached maximum iterations ({max_iterations}). Stopping.")
                    break
                
                # Sleep until next check
                logger.info(f"Sleeping for {interval_seconds} seconds until next check...")
                time.sleep(interval_seconds)
                iteration += 1
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user (KeyboardInterrupt)")
        except Exception as e:
            logger.error(f"Unexpected error during monitoring: {e}")
        
        logger.info("Monitoring finished")

# Simple command-line execution
if __name__ == "__main__":
    # Create the tracker
    tracker = SimpleTrendTracker()
    
    # Use these keywords or customize them
    keywords = ['AI', 'machine learning', 'python', 'data science', 'technology']
    
    # Run monitoring (every hour, with default keywords)
    # To stop, press Ctrl+C
    tracker.run_basic_monitoring(
        custom_keywords=keywords,
        interval_seconds=3600,  # Check hourly
        max_iterations=None  # Run indefinitely
    )