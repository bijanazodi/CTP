import requests
from bs4 import BeautifulSoup
import pandas as pd
import pandas_ta as ta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
import schedule
import time
from pycoingecko import CoinGeckoAPI

# Initialize CoinGeckoAPI
cg = CoinGeckoAPI()

# Function to scrape crypto news from multiple sources
def scrape_crypto_news():
    news_urls = [
        'https://www.coindesk.com/',
        'https://cointelegraph.com/',
        'https://news.bitcoin.com/',
        'https://www.cryptonewsz.com/',
        'https://cryptopotato.com/',
        'https://u.today/',
        'https://cryptoslate.com/'
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

# Function to fetch historical data from CoinGecko
def fetch_crypto_data(crypto_id):
    data = cg.get_coin_market_chart_by_id(id=crypto_id, vs_currency='usd', days=30)
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Function to get list of cryptocurrencies from CoinGecko API
def get_crypto_list():
    coins = cg.get_coins_list()
    return [coin['id'] for coin in coins]

# Function to analyze data
def analyze_data():
    articles = scrape_crypto_news()
    article_titles = [title for title, link in articles]
    article_sentiments = analyze_sentiment(article_titles)
    crypto_list = get_crypto_list()

    potential_movers = []

    for (title, link), sentiment_score in zip(articles, article_sentiments):
        if sentiment_score > 0.5:  # Example threshold
            for crypto_id in crypto_list[:50]:  # Limit to the first 50 for demonstration purposes
                try:
                    data = fetch_crypto_data(crypto_id)
                    if data.empty:
                        continue
                    data['MA'] = ta.sma(data['price'], length=30)
                    
                    features = data[['price', 'MA']].dropna()
                    labels = (features['price'].shift(-1) > features['price']).astype(int)
                    
                    model = RandomForestClassifier()
                    model.fit(features, labels)
                    
                    predictions = model.predict(features)
                    
                    if predictions[-1] == 1:
                        potential_movers.append((title, link, crypto_id, sentiment_score))
                except Exception as e:
                    print(f"Failed to fetch data for {crypto_id}: {e}")

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
