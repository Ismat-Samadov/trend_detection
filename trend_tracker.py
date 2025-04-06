import pandas as pd
from pytrends.request import TrendReq
import time
import datetime
import os
import logging
import sys

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

class MinimalTrendTracker:
    def __init__(self):
        """Initialize the tracker with minimal configuration"""
        # Simple initialization
        self.pytrends = TrendReq(hl='en-US', tz=360)
        self.output_dir = "trend_data"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
    
    def save_to_csv(self, df, name):
        """Save DataFrame to CSV with timestamp"""
        if df is None or df.empty:
            logger.warning(f"No data to save for {name}")
            return None
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        df.to_csv(filepath)
        logger.info(f"Saved {name} data to {filepath}")
        return filepath
    
    def track_keywords(self, keywords, timeframe='now 7-d'):
        """
        Track specific keywords and save interest over time
        
        Args:
            keywords: List of keywords to track (max 5)
            timeframe: Time period to analyze
        """
        if not keywords:
            logger.warning("No keywords provided")
            return False
            
        # Limit to 5 keywords (Google Trends limitation)
        keywords = keywords[:5]
        logger.info(f"Tracking keywords: {keywords}")
        
        try:
            # Build the payload
            self.pytrends.build_payload(
                keywords, 
                cat=0, 
                timeframe=timeframe
            )
            
            # Get interest over time (this is the most reliable API call)
            interest_df = self.pytrends.interest_over_time()
            
            if interest_df is None or interest_df.empty:
                logger.warning("No interest data returned for keywords")
                return False
                
            # Success - save to CSV
            self.save_to_csv(interest_df, "keyword_interest")
            
            # Try to get interest by region (this sometimes works)
            try:
                region_df = self.pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True)
                if region_df is not None and not region_df.empty:
                    self.save_to_csv(region_df, "interest_by_region")
            except Exception as e:
                logger.warning(f"Could not get regional data: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error tracking keywords: {e}")
            return False
    
    def get_related_topics(self, keyword):
        """Get related topics for a single keyword"""
        try:
            logger.info(f"Getting related topics for: {keyword}")
            
            # Build the payload for a single keyword
            self.pytrends.build_payload([keyword], cat=0, timeframe='today 12-m')
            
            # Get related topics
            related_topics = self.pytrends.related_topics()
            
            if keyword in related_topics:
                # Try to save "top" topics
                if 'top' in related_topics[keyword] and related_topics[keyword]['top'] is not None:
                    top_df = related_topics[keyword]['top']
                    self.save_to_csv(top_df, f"related_topics_{keyword}")
                
                # Try to save "rising" topics
                if 'rising' in related_topics[keyword] and related_topics[keyword]['rising'] is not None:
                    rising_df = related_topics[keyword]['rising']
                    self.save_to_csv(rising_df, f"rising_topics_{keyword}")
                    
                return True
            else:
                logger.warning(f"No related topics found for {keyword}")
                return False
                
        except Exception as e:
            logger.error(f"Error getting related topics for {keyword}: {e}")
            return False
    
    def run_tracking(self, keyword_sets, interval_hours=1, max_runs=None):
        """
        Run tracking for multiple keyword sets
        
        Args:
            keyword_sets: List of keyword sets (each set max 5 keywords)
            interval_hours: Hours between runs
            max_runs: Maximum number of runs (None for unlimited)
        """
        if not keyword_sets:
            logger.error("No keyword sets provided")
            return
            
        # Validate keyword sets
        validated_sets = []
        for i, keyword_set in enumerate(keyword_sets):
            if not keyword_set:
                continue
                
            # Trim to 5 keywords
            if len(keyword_set) > 5:
                logger.warning(f"Keyword set {i+1} has more than 5 keywords; trimming to first 5")
                keyword_set = keyword_set[:5]
                
            validated_sets.append(keyword_set)
            
        if not validated_sets:
            logger.error("No valid keyword sets after validation")
            return
            
        logger.info(f"Starting tracking with {len(validated_sets)} keyword sets")
        for i, keywords in enumerate(validated_sets):
            logger.info(f"Set {i+1}: {', '.join(keywords)}")
            
        # Convert hours to seconds
        interval_seconds = interval_hours * 3600
        run_count = 0
        
        try:
            while True:
                run_count += 1
                logger.info(f"Starting run #{run_count}")
                
                # Process each keyword set
                for i, keywords in enumerate(validated_sets):
                    logger.info(f"Processing keyword set {i+1}: {', '.join(keywords)}")
                    
                    # Track this set of keywords
                    success = self.track_keywords(keywords)
                    
                    # If we successfully tracked keywords, also try to get related topics
                    # for the first keyword in the set (to avoid too many API calls)
                    if success and keywords:
                        # Add a short delay to avoid hitting rate limits
                        time.sleep(3)
                        self.get_related_topics(keywords[0])
                    
                    # Add a delay between keyword sets
                    if i < len(validated_sets) - 1:
                        logger.info("Waiting 5 seconds before next keyword set...")
                        time.sleep(5)
                
                # Check if we've reached the maximum number of runs
                if max_runs and run_count >= max_runs:
                    logger.info(f"Reached maximum runs ({max_runs}). Stopping.")
                    break
                    
                # Wait for the next interval
                next_run_time = datetime.datetime.now() + datetime.timedelta(seconds=interval_seconds)
                logger.info(f"Run completed. Next run at: {next_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"Sleeping for {interval_hours} hours...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Tracking stopped by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"Unexpected error during tracking: {e}")
            
        logger.info("Tracking finished")

def main():
    """Main function to run the tracker"""
    # Create tracker
    tracker = MinimalTrendTracker()
    
    # Define your keyword sets here (each set limited to 5 keywords)
    keyword_sets = [
        ['AI', 'machine learning', 'data science'],
        ['python', 'javascript', 'programming'],
        ['bitcoin', 'ethereum', 'cryptocurrency'],
        ['mobile apps', 'software development', 'tech startups'],
    ]
    
    # Set your tracking interval (in hours)
    interval_hours = 1
    
    # Set maximum runs (None for unlimited)
    max_runs = None
    
    # Run the tracker
    tracker.run_tracking(
        keyword_sets=keyword_sets,
        interval_hours=interval_hours,
        max_runs=max_runs
    )

if __name__ == "__main__":
    main()