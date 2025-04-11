import finnhub
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import yfinance as yf
import plotly.express as px
from scipy import stats

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError as e:
    print(f"Downloading missing NLTK data: {str(e)}")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# Initialize Finnhub client - you'll need to replace this with your API key
finnhub_client = finnhub.Client(api_key="cvr0hfhr01qp88cnv9dgcvr0hfhr01qp88cnv9e0")

def analyze_sentiment(text):
    """
    Analyze sentiment of text and return score from 1-10
    """
    analysis = TextBlob(text)
    # Convert polarity (-1 to 1) to scale of 1-10
    sentiment_score = ((analysis.sentiment.polarity + 1) * 4.5) + 1
    return round(sentiment_score, 1)

def extract_keywords(text, top_n=5):
    """
    Extract top 5 keywords from text using TF-IDF
    """
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    
    if not tokens:
        return []
    
    # Create TF-IDF vector
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([' '.join(tokens)])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Get top keywords
        top_indices = scores.argsort()[-top_n:][::-1]
        return [feature_names[i] for i in top_indices]
    except:
        return tokens[:top_n]  # Fallback to simple frequency

def categorize_news(text):
    """
    Categorize news into predefined categories
    """
    categories = {
        'Earnings': ['earnings', 'revenue', 'profit', 'loss', 'quarterly', 'financial results'],
        'Management': ['ceo', 'executive', 'management', 'board', 'leadership'],
        'Product': ['product', 'launch', 'release', 'innovation', 'development'],
        'Market': ['market', 'stock', 'shares', 'trading', 'investors'],
        'Merger & Acquisition': ['acquisition', 'merger', 'deal', 'buyout', 'takeover'],
        'Regulatory': ['regulation', 'compliance', 'legal', 'lawsuit', 'sec'],
        'Technology': ['technology', 'tech', 'digital', 'innovation', 'software'],
        'Competition': ['competitor', 'competition', 'market share', 'industry'],
        'Strategy': ['strategy', 'plan', 'growth', 'expansion', 'restructuring'],
        'Economic': ['economy', 'economic', 'inflation', 'gdp', 'recession']
    }
    
    text_lower = text.lower()
    category_scores = {}
    
    for category, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        category_scores[category] = score
    
    # Return top matching category
    if any(category_scores.values()):
        return max(category_scores.items(), key=lambda x: x[1])[0]
    return 'Other'

def get_stock_news(symbol='AAPL', from_date=None, to_date=None):
    """
    Get news for a specific stock symbol with support for longer date ranges
    """
    if from_date is None:
        from_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    if to_date is None:
        to_date = datetime.now().strftime('%Y-%m-%d')
    
    # Convert string dates to datetime objects for manipulation
    from_dt = datetime.strptime(from_date, '%Y-%m-%d')
    to_dt = datetime.strptime(to_date, '%Y-%m-%d')
    
    all_news = []
    current_from = from_dt
    
    # Calculate total number of chunks needed
    total_days = (to_dt - from_dt).days
    num_chunks = (total_days // 30) + (1 if total_days % 30 > 0 else 0)
    current_chunk = 0
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process in 30-day chunks to handle API limitations
    while current_from <= to_dt:
        current_chunk += 1
        current_to = min(current_from + timedelta(days=30), to_dt)
        
        status_text.text(f"Fetching news chunk {current_chunk}/{num_chunks} ({current_from.strftime('%Y-%m-%d')} to {current_to.strftime('%Y-%m-%d')})")
        progress_bar.progress(current_chunk / num_chunks)
        
        try:
            chunk_news = finnhub_client.company_news(
                symbol,
                _from=current_from.strftime('%Y-%m-%d'),
                to=current_to.strftime('%Y-%m-%d')
            )
            all_news.extend(chunk_news)
            
        except Exception as e:
            st.error(f"Error fetching news for period {current_from.strftime('%Y-%m-%d')} to {current_to.strftime('%Y-%m-%d')}: {str(e)}")
        
        # Move to next chunk
        current_from = current_to + timedelta(days=1)
    
    # Clean up progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Convert to DataFrame and process
    df = pd.DataFrame(all_news)
    if not df.empty:
        # Add sentiment analysis
        df['sentiment_score'] = df['summary'].apply(analyze_sentiment)
        # Add keywords
        df['keywords'] = df['summary'].apply(extract_keywords)
        # Add categories
        df['category'] = df['summary'].apply(categorize_news)
        # Sort by datetime to ensure chronological order
        df = df.sort_values('datetime', ascending=True)
        
    return df

def get_stock_prices(symbol, start_date, end_date):
    """
    Get stock prices for the specified period
    """
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        # Convert index to column instead of keeping as index
        df = df.reset_index()
        df['Date'] = df['Date'].dt.date  # Convert datetime to date
        return df
    except Exception as e:
        st.error(f"Error fetching stock prices: {str(e)}")
        return pd.DataFrame()

def display_news():
    st.title("Stock News Tracker")
    
    # Sidebar for user input
    with st.sidebar:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL").upper()
        
        # Add date inputs with new defaults
        today = datetime.now()
        default_start = today - timedelta(days=10)
        
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            max_value=today
        )
        
        end_date = st.date_input(
            "End Date",
            value=today,
            max_value=today
        )
        
        # Add source filter
        if 'news_df' in st.session_state:
            available_sources = sorted(st.session_state.news_df['source'].unique())
            selected_sources = st.multiselect(
                "Filter News Sources",
                options=available_sources,
                default=available_sources
            )
        else:
            selected_sources = []
        
        # Add refresh button
        refresh = st.button("Run Analysis")
        
        st.info("Please ensure you have set your Finnhub API key in the code.")
    
    # Get news data only when refresh button is clicked or on initial load
    if 'news_df' not in st.session_state or refresh:
        news_df = get_stock_news(
            symbol,
            from_date=start_date.strftime('%Y-%m-%d'),
            to_date=end_date.strftime('%Y-%m-%d')
        )
        # Get stock prices
        price_df = get_stock_prices(symbol, start_date, end_date)
        
        # Store both dataframes in session state
        st.session_state.news_df = news_df
        st.session_state.price_df = price_df
    else:
        news_df = st.session_state.news_df
        price_df = st.session_state.price_df

    if not news_df.empty:
        # Filter news based on selected sources
        if selected_sources:
            news_df = news_df[news_df['source'].isin(selected_sources)]

        # Create tabs
        tab1, tab2 = st.tabs(["News Analysis", "Sentiment-Price Correlation"])
        
        with tab1:
            st.subheader(f"Latest News for {symbol}")
            
            # Display key metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of News", len(news_df))
            with col2:
                avg_sentiment = round(news_df['sentiment_score'].mean(), 2)
                st.metric("Average Sentiment", f"{avg_sentiment}/10")
            with col3:
                date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                st.metric("Date Range", date_range)

            # Add sentiment distribution visualization with five bars
            st.subheader("Sentiment Distribution")
            
            # Create bins for sentiment scores
            bins = [0, 2, 4, 6, 8, 10]
            labels = ['0-2', '2-4', '4-6', '6-8', '8-10']
            news_df['sentiment_range'] = pd.cut(news_df['sentiment_score'], bins=bins, labels=labels, include_lowest=True)
            sentiment_data = pd.DataFrame({
                'Sentiment Range': news_df['sentiment_range'].value_counts().sort_index()
            })
            
            # Use a container with custom width
            with st.container():
                st.bar_chart(sentiment_data, use_container_width=True)

            # Create a clean table view of the news with date
            news_table = pd.DataFrame({
                'Date': news_df['datetime'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M')),
                'Headline': news_df['headline'],
                'Sentiment': news_df['sentiment_score'].apply(lambda x: f"{x:.1f}/10"),
                'Relative Sentiment': news_df['sentiment_score'].apply(lambda x: f"{((x / avg_sentiment) - 1) * 10:.1f}"),
                'Category': news_df['category'],
                'Source': news_df['source']
            })
            
            # Display the table with custom formatting
            st.markdown("""
                <style>
                    .stTable {
                        width: 100% !important;
                        font-size: 80% !important;
                    }
                    .dataframe {
                        width: 100% !important;
                        font-size: 80% !important;
                    }
                    .st-emotion-cache-1n76uvr {
                        font-size: 80% !important;
                    }
                    .st-emotion-cache-1inwz65 {
                        font-size: 80% !important;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            st.table(news_table.style.set_properties(**{
                'text-align': 'left',
                'white-space': 'normal',
                'height': 'auto',
                'max-width': 'none'
            }))
            
            # Add detailed view option
            st.subheader("Detailed News View")
            for _, row in news_df.iterrows():
                with st.expander(f"Details for: {row['headline']}"):
                    st.write(f"**Keywords:** {', '.join(row['keywords'])}")
                    st.write(f"**Summary:** {row['summary']}")
                    st.write(f"**Date:** {datetime.fromtimestamp(row['datetime']).strftime('%Y-%m-%d %H:%M')}")
                    if row['url']:
                        st.markdown(f"[Read more]({row['url']})")
        
        with tab2:
            st.subheader("Sentiment-Price Correlation Analysis")
            
            if not price_df.empty:
                # Calculate next day's price change percentage
                price_df['Next_Day_Change'] = price_df['Close'].shift(-1).pct_change() * 100
                
                # Prepare data for correlation analysis
                news_df['date'] = news_df['datetime'].apply(lambda x: datetime.fromtimestamp(x).date())
                news_daily = news_df.groupby('date')['sentiment_score'].agg(['mean', 'count']).reset_index()
                news_daily.columns = ['Date', 'Avg_Sentiment', 'News_Count']
                
                # Calculate relative sentiment
                avg_sentiment = news_daily['Avg_Sentiment'].mean()
                news_daily['Relative_Sentiment'] = news_daily['Avg_Sentiment'].apply(lambda x: ((x / avg_sentiment) - 1) * 10)
                
                # Merge with price data - now both DataFrames have Date as a column
                merged_df = pd.merge(news_daily, price_df[['Date', 'Next_Day_Change']], on='Date', how='inner')
                # Remove the last row since it won't have next day's change
                merged_df = merged_df.dropna(subset=['Next_Day_Change'])
                
                if not merged_df.empty:
                    # Calculate correlation
                    correlation, p_value = stats.pearsonr(merged_df['Relative_Sentiment'], merged_df['Next_Day_Change'])
                    
                    # Display correlation metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Correlation Coefficient", f"{correlation:.3f}")
                    with col2:
                        st.metric("P-value", f"{p_value:.3f}")
                    
                    # Create scatter plot
                    fig = px.scatter(merged_df, 
                                   x='Relative_Sentiment', 
                                   y='Next_Day_Change',
                                   title=f'Relative Sentiment vs Next Day Price Change ({symbol})',
                                   labels={'Relative_Sentiment': 'Relative Sentiment Score (-10 to +10)',
                                          'Next_Day_Change': 'Next Day Price Change (%)'},
                                   trendline="ols")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display data table
                    st.subheader("Correlation Data")
                    correlation_table = merged_df[['Date', 'Relative_Sentiment', 'Next_Day_Change', 'News_Count']]
                    correlation_table.columns = ['Date', 'Relative Sentiment', 'Next Day Price Change (%)', 'Number of News']
                    st.dataframe(correlation_table.style.format({
                        'Relative Sentiment': '{:.2f}',
                        'Next Day Price Change (%)': '{:.2f}'
                    }))
                else:
                    st.warning("Insufficient data for correlation analysis.")
            else:
                st.warning("Unable to fetch stock price data for correlation analysis.")
    else:
        st.warning("No news found for the specified symbol.")

if __name__ == "__main__":
    display_news()