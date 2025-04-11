import finnhub
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
from .sentiment_analysis import analyze_sentiment, extract_keywords, categorize_news

finnhub_client = finnhub.Client(api_key="cvr0hfhr01qp88cnv9dgcvr0hfhr01qp88cnv9e0")

def get_stock_news(symbol='AAPL', from_date=None, to_date=None):
    """Get news for a specific stock symbol with support for longer date ranges"""
    if from_date is None:
        from_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    if to_date is None:
        to_date = datetime.now().strftime('%Y-%m-%d')
    
    from_dt = datetime.strptime(from_date, '%Y-%m-%d')
    to_dt = datetime.strptime(to_date, '%Y-%m-%d')
    
    all_news = []
    current_from = from_dt
    
    total_days = (to_dt - from_dt).days
    num_chunks = (total_days // 30) + (1 if total_days % 30 > 0 else 0)
    current_chunk = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while current_from <= to_dt:
        current_chunk += 1
        current_to = min(current_from + timedelta(days=30), to_dt)
        
        status_text.text(f"Fetching news chunk {current_chunk}/{num_chunks}")
        progress_bar.progress(current_chunk / num_chunks)
        
        try:
            chunk_news = finnhub_client.company_news(
                symbol,
                _from=current_from.strftime('%Y-%m-%d'),
                to=current_to.strftime('%Y-%m-%d')
            )
            all_news.extend(chunk_news)
            
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
        
        current_from = current_to + timedelta(days=1)
    
    progress_bar.empty()
    status_text.empty()
    
    df = pd.DataFrame(all_news)
    if not df.empty:
        df['sentiment_score'] = df['summary'].apply(analyze_sentiment)
        df['keywords'] = df['summary'].apply(extract_keywords)
        df['category'] = df['summary'].apply(categorize_news)
        df = df.sort_values('datetime', ascending=True)
        
    return df

def get_stock_prices(symbol, start_date, end_date):
    """Get stock prices for the specified period"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        df = df.reset_index()
        df['Date'] = df['Date'].dt.date
        return df
    except Exception as e:
        st.error(f"Error fetching stock prices: {str(e)}")
        return pd.DataFrame()
