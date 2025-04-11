# Stock News Sentiment Analysis

Analyze stock news sentiment and correlate it with stock price movements using Finnhub API and machine learning.

## Features
- Real-time stock news analysis
- Sentiment scoring (1-10 scale)
- Keyword extraction using TF-IDF
- News categorization
- Price-sentiment correlation
- Interactive Streamlit dashboard

## Prerequisites
- Python 3.8+
- Finnhub API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Sentiment-Analysis-and-Stock-Correlation.git
cd Sentiment-Analysis-and-Stock-Correlation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add your Finnhub API key:
```
FINNHUB_API_KEY=your_api_key_here
```

## Usage
Run the Streamlit app:
```bash
streamlit run app.py
```

## Project Structure
```
├── utils/
│   ├── sentiment_analysis.py
│   └── stock_data.py
├── app.py
├── requirements.txt
└── README.md
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first.

## License
MIT
