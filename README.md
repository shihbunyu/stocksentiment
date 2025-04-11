# Stock Sentiment Analysis

A Python application that analyzes stock market sentiment using news data and correlates it with price movements.

## Features
- News sentiment analysis
- Keyword extraction
- News categorization
- Price-sentiment correlation
- Interactive visualization

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure Finnhub API key in `utils/stock_data.py`

3. Run the application:
```bash
streamlit run app.py
```

## Structure
- `app.py`: Main Streamlit application
- `utils/`: Utility functions
  - `sentiment_analysis.py`: Text analysis functions
  - `stock_data.py`: Stock data retrieval
