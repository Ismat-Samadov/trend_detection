# Trend Detection

A Python tool for tracking Google search trends and saving the data to CSV files for analysis.

## Overview

This project allows you to monitor interest over time for specific keywords and topics using Google Trends data. The tool periodically fetches data for configured keyword sets and saves the results as timestamped CSV files, making it easy to track changes in search interest and discover related topics.

## Features

- **Keyword Interest Tracking**: Monitor search volume trends for up to 5 keywords at a time
- **Multiple Keyword Sets**: Define different groups of keywords to track diverse topics
- **Geographical Data**: Collect interest data broken down by country
- **Related Topics**: Discover both top and rising topics related to your keywords
- **Scheduled Monitoring**: Run at configurable intervals (hourly, daily, etc.)
- **CSV Data Storage**: All data is saved as timestamped CSV files for easy analysis
- **Detailed Logging**: Comprehensive logging of all operations and errors

## Requirements

- Python 3.6+
- pandas
- pytrends (Unofficial Google Trends API)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ismat-Samadov/trend_detection.git
cd trend_detection
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the script with default settings:

```bash
python minimal_trends.py
```

This will track the default keyword sets with hourly updates.

### Configuration

Edit the `main()` function in the script to customize:

```python
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
```

### Output

The script creates a `trend_data` directory where it saves all CSV files:

- `keyword_interest_YYYYMMDD_HHMMSS.csv`: Interest over time for each keyword
- `interest_by_region_YYYYMMDD_HHMMSS.csv`: Interest by country
- `related_topics_KEYWORD_YYYYMMDD_HHMMSS.csv`: Top related topics
- `rising_topics_KEYWORD_YYYYMMDD_HHMMSS.csv`: Rising related topics

### Logging

A detailed log is written to `trend_tracker.log` in the project directory.

## How It Works

1. **Initialization**: The tracker creates necessary directories and initializes the Google Trends client
2. **Keyword Processing**: For each set of keywords (up to 5 per set):
   - Collects interest over time data
   - Collects geographical distribution data
   - Collects related topics for the primary keyword
3. **Data Storage**: All data is saved to timestamped CSV files
4. **Scheduling**: The process repeats at the configured interval

## API Limitations

Note that the Google Trends API has certain limitations:

- Maximum of 5 keywords per search
- Rate limiting (hence the delays between requests)
- Some endpoints may be unstable or return empty results
- Data is normalized and relative, not absolute search volumes

## Advanced Usage

### Changing Time Periods

Modify the `timeframe` parameter in `track_keywords()` to change the analysis period:

```python
# Examples:
timeframe='now 1-d'    # Last day
timeframe='now 7-d'    # Last 7 days
timeframe='today 3-m'  # Last 3 months
timeframe='today 12-m' # Last 12 months
```

### Adjusting Geographical Focus

To focus on trends for a specific country, modify the `geo` parameter in `pytrends.build_payload()`:

```python
self.pytrends.build_payload(
    keywords, 
    cat=0, 
    timeframe=timeframe,
    geo='US'  # Country code for USA
)
```

## Troubleshooting

### Common Issues

1. **404 Errors**: The Google Trends API sometimes returns 404 errors. The script handles these gracefully and continues with other keyword sets.

2. **Empty Results**: Some keywords may return empty datasets. This is normal for terms with very low search volume.

3. **Rate Limiting**: If you see connection errors, try increasing the delay between requests by modifying the `time.sleep()` values.

4. **Index Errors**: These can occur when processing related queries. The script includes error handling for these cases.

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

This project uses the unofficial Google Trends API ([pytrends](https://github.com/GeneralMills/pytrends)) to access trend data.