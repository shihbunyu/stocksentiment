import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import the trading strategy functions
from trading_strategy import (
    get_stock_data,
    calculate_moving_averages,
    identify_golden_cross,
    implement_strategy,
    analyze_results
)

def create_price_chart(data, positions):
    fig = go.Figure()
    
    # Add price data
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Stock Price',
        line=dict(color='blue')
    ))
    
    # Add moving averages
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MA50'],
        name='50-day MA',
        line=dict(color='orange')
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MA200'],
        name='200-day MA',
        line=dict(color='red')
    ))
    
    # Add buy points
    if not positions.empty:
        fig.add_trace(go.Scatter(
            x=positions['BuyDate'],
            y=positions['BuyPrice'],
            mode='markers',
            name='Buy Points',
            marker=dict(color='green', size=10)
        ))
        
        # Add sell points
        fig.add_trace(go.Scatter(
            x=positions['SellDate'],
            y=positions['SellPrice'],
            mode='markers',
            name='Sell Points',
            marker=dict(color='red', size=10)
        ))
    
    fig.update_layout(
        title='Stock Price with Moving Averages and Trading Signals',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600
    )
    
    return fig

def create_profit_distribution(positions):
    if positions.empty:
        st.warning("No trades found for the selected period.")
        return None
    
    plt.figure(figsize=(10, 6))
    plt.hist(positions['ProfitPct'], bins=30, edgecolor='black')
    plt.title('Distribution of Trade Profits')
    plt.xlabel('Profit (%)')
    plt.ylabel('Number of Trades')
    return plt.gcf()

def main():
    st.set_page_config(page_title="Stock Trading Strategy Visualizer", layout="wide")
    st.title("Stock Trading Strategy Visualizer")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'positions' not in st.session_state:
        st.session_state.positions = pd.DataFrame()
    
    # Sidebar for user inputs
    st.sidebar.header("Strategy Parameters")
    ticker = st.sidebar.text_input("Stock Ticker", "MSFT")
    period = st.sidebar.selectbox("Time Period", ["1y", "5y", "10y", "max"])
    
    try:
        if st.sidebar.button("Run Analysis"):
            with st.spinner("Running analysis..."):
                # Get stock data
                st.session_state.data = get_stock_data(ticker, period)
                
                # Calculate moving averages
                st.session_state.data = calculate_moving_averages(st.session_state.data)
                
                # Identify golden cross
                st.session_state.data = identify_golden_cross(st.session_state.data)
                
                # Implement strategy
                st.session_state.positions = implement_strategy(st.session_state.data)
                
                # Analyze results
                results = analyze_results(st.session_state.positions)
        
        # Only show results if we have data
        if st.session_state.data is not None:
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["Price Chart", "Trade Statistics", "Detailed Trades"])
            
            with tab1:
                st.plotly_chart(create_price_chart(st.session_state.data, st.session_state.positions), use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Performance Metrics")
                    if not st.session_state.positions.empty:
                        total_trades = len(st.session_state.positions)
                        win_trades = len(st.session_state.positions[st.session_state.positions['ProfitPct'] > 0])
                        win_rate = win_trades / total_trades * 100
                        avg_profit = st.session_state.positions['ProfitPct'].mean()
                        
                        st.metric("Total Trades", total_trades)
                        st.metric("Win Rate", f"{win_rate:.2f}%")
                        st.metric("Average Profit", f"{avg_profit:.2f}%")
                    else:
                        st.warning("No trades found for the selected period.")
                
                with col2:
                    st.subheader("Profit Distribution")
                    fig = create_profit_distribution(st.session_state.positions)
                    if fig is not None:
                        st.pyplot(fig)
            
            with tab3:
                st.subheader("Detailed Trade History")
                if not st.session_state.positions.empty:
                    st.dataframe(st.session_state.positions)
                else:
                    st.warning("No trades found for the selected period.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try a different stock ticker or time period.")

if __name__ == "__main__":
    main() 