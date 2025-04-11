import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from scipy import stats
import plotly.express as px
import nltk
from utils.stock_data import get_stock_news, get_stock_prices

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except LookupError as e:
    print(f"Downloading missing NLTK data: {str(e)}")

def display_news():
    st.title("Stock Sentiment Analysis")
    st.write("Welcome to the Stock Sentiment Analysis tool")

if __name__ == "__main__":
    display_news()
