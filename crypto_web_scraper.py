import requests
from bs4 import BeautifulSoup
import pandas as pd
import pandas_ta as ta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
import schedule
import time
import yfinance as yf

# Function to scrape crypto news from multiple sources
def scrape_crypto_news():
    news_urls = [
        'https://www.coindesk.com/',
        'https://cointelegraph.com/',
        'https://news.bitcoin.com/',
        'https://www.cryptonewsz.com/'
    ]

    articles = []
    for url in news_urls:
        response = requests.get(url)
        print(f"Scraping {url}, response status code: {response.status_code}")
        soup = BeautifulSoup(response.content, 'html.parser')

        for article in soup.select('h4 a')[:10]:  # Adjust selector based on the website's structure
            title = article.text.strip()
            link = article['href']
            if not link.startswith('http'):
                link = url + link
            articles.append((title, link))

    print(f"Scraped articles: {articles}")
    return articles

# Function to perform sentiment analysis
def analyze_sentiment(texts):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [analyzer.polarity_scores(text)['compound'] for text in texts]
    return sentiment_scores

# Function to fetch historical data
def fetch_crypto_data(crypto_id):
    data = yf.download(crypto_id, period='30d', interval='1d')
    data.reset_index(inplace=True)
    return data

# Function to analyze data
def analyze_data():
    articles = scrape_crypto_news()

    article_titles = [title for title, link in articles]
    article_sentiments = analyze_sentiment(article_titles)

    potential_movers = []

    for (title, link), sentiment_score in zip(articles, article_sentiments):
        if sentiment_score > 0.5:  # Example threshold
            # Dynamically identify crypto IDs from titles (here we are using a placeholder list of crypto IDs)
            crypto_ids = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOGE-USD', 'XRP-USD']  # Replace with actual detection
            for crypto_id in crypto_ids:
                data = fetch_crypto_data(crypto_id)
                data['MA'] = ta.sma(data['Close'], length=30)
                
                features = data[['Close', 'MA']].dropna()
                labels = (features['Close'].shift(-1) > features['Close']).astype(int)
                
                model = RandomForestClassifier()
                model.fit(features, labels)
                
                predictions = model.predict(features)
                
                if predictions[-1] == 1:
                    potential_movers.append((title, link, crypto_id, sentiment_score))

    print(f"Potential movers: {potential_movers}")
    return potential_movers

# Scheduler to run the job
def job():
    movers = analyze_data()
    print("Potential Future Top Movers:", movers)

schedule.every().day.at("10:00").do(job)

# Run immediately for debugging
job()

while True:
    schedule.run_pending()
    time.sleep(1)
