from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def analyze_sentiment(text):
    """Analyze sentiment of text and return score from 1-10"""
    analysis = TextBlob(text)
    sentiment_score = ((analysis.sentiment.polarity + 1) * 4.5) + 1
    return round(sentiment_score, 1)

def extract_keywords(text, top_n=5):
    """Extract top 5 keywords from text using TF-IDF"""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    
    if not tokens:
        return []
    
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([' '.join(tokens)])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        top_indices = scores.argsort()[-top_n:][::-1]
        return [feature_names[i] for i in top_indices]
    except:
        return tokens[:top_n]

def categorize_news(text):
    """Categorize news into predefined categories"""
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
    
    return max(category_scores.items(), key=lambda x: x[1])[0] if any(category_scores.values()) else 'Other'
