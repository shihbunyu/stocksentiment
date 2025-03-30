# Stock Trading Strategy Visualizer

This application visualizes a Golden Cross trading strategy for stocks. The strategy identifies buy signals when the 50-day moving average crosses above the 200-day moving average, and sells when either:
1. A 15% profit target is reached
2. The maximum holding period of 60 days is reached

## Features

- Interactive stock price chart with moving averages
- Buy and sell points visualization
- Performance metrics dashboard
- Profit distribution analysis
- Detailed trade history
- Customizable stock ticker and time period

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the sidebar to:
   - Enter a stock ticker symbol (default: MSFT)
   - Select the time period for analysis
   - Click "Run Analysis" to update the results

4. Explore the results using the three tabs:
   - Price Chart: Interactive chart showing price, moving averages, and trade signals
   - Trade Statistics: Performance metrics and profit distribution
   - Detailed Trades: Complete list of all trades with their details

## Strategy Details

The trading strategy is based on the Golden Cross pattern:
- Buy Signal: When the 50-day moving average crosses above the 200-day moving average
- Sell Conditions:
  - Target reached: 15% profit
  - Maximum holding period: 60 days 