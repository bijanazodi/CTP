import requests
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline
from pycoingecko import CoinGeckoAPI
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize CoinGeckoAPI and sentiment analysis pipeline
cg = CoinGeckoAPI()
sentiment_pipeline = pipeline('sentiment-analysis')

def fetch_coingecko_data(crypto_id, days=10):
    try:
        data = cg.get_coin_market_chart_by_id(id=crypto_id, vs_currency='usd', days=days)
        prices = data['prices']
        total_volumes = data['total_volumes']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.rename(columns={'price': 'Close'}, inplace=True)
        df['High'] = df['Close']
        df['Low'] = df['Close']
        df['Open'] = df['Close']
        volume_df = pd.DataFrame(total_volumes, columns=['timestamp', 'volume'])
        volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
        volume_df.set_index('timestamp', inplace=True)
        df['Volume'] = volume_df['volume']
        return df
    except Exception as e:
        logging.error(f"Error fetching CoinGecko data for {crypto_id}: {e}")
        return None

def calculate_indicators(data):
    data['MA'] = ta.sma(data['Close'], length=10)
    data['RSI'] = ta.rsi(data['Close'], length=14)
    macd = ta.macd(data['Close'])
    if macd is not None and not macd.empty:
        data['MACD'] = macd['MACD_12_26_9']
        data['MACD_signal'] = macd['MACDs_12_26_9']
        data['MACD_diff'] = macd['MACDh_12_26_9']
    bollinger = ta.bbands(data['Close'], length=20, std=2)
    if (bollinger is not None) and not bollinger.empty:
        data['BB_upper'] = bollinger['BBU_20_2.0']
        data['BB_lower'] = bollinger['BBL_20_2.0']
    data['Support'] = ta.sma(data['Close'], length=30)
    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
    data['Stoch'] = ta.stoch(data['High'], data['Low'], data['Close'])['STOCHk_14_3_3']
    data['OBV'] = ta.obv(data['Close'], data['Volume'])
    return data

def prepare_data_lstm(data):
    features = data[['Close', 'MA', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'BB_upper', 'BB_lower', 'Support', 'ATR', 'Stoch', 'OBV']].dropna()
    labels = features['Close'].shift(-1).dropna()
    features = features.iloc[:-1]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    X = []
    y = []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    return X, y, scaler, features.columns.tolist(), scaled_data

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def analyze_crypto(crypto_id, crypto_symbol, latest_price, days=10, interval='hourly'):
    try:
        logging.info(f"Fetching data for {crypto_id}...")

        # Skip Binance, directly fetch data from CoinGecko
        data = fetch_coingecko_data(crypto_id, days)
        
        if data is None or data.empty:
            raise ValueError(f"Failed to fetch data for {crypto_id}.")

        logging.info(f"Calculating indicators for {crypto_id}...")
        data.loc[data.index[-1], 'Close'] = latest_price
        data = calculate_indicators(data)

        logging.info(f"Preparing LSTM data for {crypto_id}...")
        X, y, scaler, features, scaled_data = prepare_data_lstm(data)

        logging.info(f"Building and training LSTM model for {crypto_id}...")
        model = build_lstm_model((X.shape[1], X.shape[2]))
        early_stopping = EarlyStopping(monitor='loss', patience=3)
        checkpoint = ModelCheckpoint(f'models/{crypto_id}_best_model.h5', save_best_only=True, monitor='loss', mode='min')
        model.fit(X, y, epochs=20, batch_size=16, callbacks=[early_stopping, checkpoint], verbose=1)

        logging.info(f"Making predictions with LSTM for {crypto_id}...")
        last_60_days = data[-60:][features].values
        last_60_days_scaled = scaler.transform(last_60_days)
        X_test = []
        X_test.append(last_60_days_scaled)
        X_test = np.array(X_test)

        predicted_price_scaled = model.predict(X_test)
        predicted_price = scaler.inverse_transform(np.concatenate((predicted_price_scaled, np.zeros((predicted_price_scaled.shape[0], scaled_data.shape[1]-1))), axis=1))[:,0]

        # Calculate RMSE and MAE for model performance
        y_pred = model.predict(X)
        y_true = y
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # Calculate percentage change for profit potential
        profit_potential = ((predicted_price[0] - latest_price) / latest_price) * 100

        # Determine sentiment based on percentage change
        if profit_potential < -10:
            next_period_prediction = "Lowest"
        elif -10 <= profit_potential < 0:
            next_period_prediction = "Low"
        elif 0 <= profit_potential < 10:
            next_period_prediction = "High"
        else:
            next_period_prediction = "Highest"

        # Calculate optimal buy and sell prices based on prediction
        optimal_sell_price = predicted_price[0]
        if next_period_prediction in ["High", "Highest"]:
            buy_price = latest_price * 0.98
        else:
            recent_support = data['Support'].iloc[-1]
            buy_price = min(recent_support, latest_price * 0.95)

        historical_prices = data['Close'].pct_change().dropna().tolist()
        sentiment_text = ' '.join(map(str, historical_prices))[:512]  # Truncate to fit the model's max length
        sentiment_text = [sentiment_text]  # Convert to a list of strings

        logging.info(f"Analyzing sentiment for {crypto_id}...")
        sentiment_result = sentiment_pipeline(sentiment_text)
        sentiment_score = sum([r['score'] if r['label'] == 'POSITIVE' else -r['score'] for r in sentiment_result])
        long_term_sentiment = "Positive" if sentiment_score > 0 else "Negative"

        return {
            "crypto_id": crypto_id,
            "crypto_symbol": crypto_symbol,
            "latest_price": latest_price,
            "predicted_price": predicted_price[0],
            "next_period_prediction": next_period_prediction,
            "optimal_buy_price": f"{buy_price:.8f}",
            "optimal_sell_price": f"{optimal_sell_price:.8f}",
            "profit_potential": profit_potential,
            "historical_data_summary": {
                "start_date": str(data.index[0]),
                "end_date": str(data.index[-1]),
                "initial_close_price": f"{data['Close'].iloc[0]:.8f}",
                "final_close_price": f"{data['Close'].iloc[-1]:.8f}",
                "average_close_price": f"{data['Close'].mean():.8f}"
            },
            "long_term_sentiment": long_term_sentiment,
            "performance": {
                "RMSE": f"{rmse:.8f}",
                "MAE": f"{mae:.8f}"
            }
        }
    except Exception as e:
        logging.error(f"Error in analyzing cryptocurrency {crypto_id}: {e}")
        return None

def get_coinbase_products():
    try:
        response = requests.get("https://api.exchange.coinbase.com/products")
        response.raise_for_status()
        products = response.json()
        usd_products = [product for product in products if product['quote_currency'] == 'USD']
        logging.info(f"Retrieved {len(usd_products)} USD trading pairs from Coinbase.")
        return usd_products
    except Exception as e:
        logging.error(f"Error fetching products from Coinbase: {e}")
        return []

def get_top_cryptos_from_coingecko(limit=200):
    try:
        coins = cg.get_coins_markets(vs_currency='usd', order='volume_desc', per_page=limit, page=1)
        return coins
    except Exception as e:
        logging.error(f"Error fetching top cryptos from CoinGecko: {e}")
        return []

def analyze_all_coinbase_cryptos():
    coinbase_products = get_coinbase_products()
    if not coinbase_products:
        print("Failed to fetch tradable cryptocurrencies from Coinbase.")
        return

    top_cryptos = get_top_cryptos_from_coingecko(200)
    if not top_cryptos:
        print("Failed to fetch top cryptocurrencies from CoinGecko.")
        return

    analysis_results = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(analyze_crypto, crypto['id'], product['base_currency'], crypto['current_price']): crypto
            for crypto in top_cryptos for product in coinbase_products if product['base_currency'].lower() == crypto['symbol']
        }
        for future in as_completed(futures):
            crypto = futures[future]
            try:
                result = future.result()
                if result:
                    analysis_results.append(result)
            except Exception as e:
                logging.error(f"Error analyzing {crypto['id']}: {e}")

    analysis_results = sorted(analysis_results, key=lambda x: x['profit_potential'], reverse=True)

    for result in analysis_results[:5]:
        print("\nCryptocurrency Analysis Result:\n")
        print(f"Cryptocurrency: {result['crypto_id']}")
        print(f"Symbol: {result['crypto_symbol'].upper()}")
        print(f"Latest Price: ${result['latest_price']:.8f}")
        print(f"Predicted Price: ${result['predicted_price']:.8f}")
        print(f"Next Period Prediction: {result['next_period_prediction']}")
        print(f"Optimal Buy Price: ${result['optimal_buy_price']}")
        print(f"Optimal Sell Price: ${result['optimal_sell_price']}")
        print(f"Profit Potential: {result['profit_potential']:.2f}%")
        print("Historical Data Summary:")
        print(f"  Start Date: {result['historical_data_summary']['start_date']}")
        print(f"  End Date: {result['historical_data_summary']['end_date']}")
        print(f"  Initial Close Price: ${result['historical_data_summary']['initial_close_price']}")
        print(f"  Final Close Price: ${result['historical_data_summary']['final_close_price']}")
        print(f"  Average Close Price: ${result['historical_data_summary']['average_close_price']}")
        print(f"Long Term Sentiment: {result['long_term_sentiment']}")
        print(f"Model Performance: RMSE = {result['performance']['RMSE']}, MAE = {result['performance']['MAE']}")

if __name__ == "__main__":
    # Uncomment this section to run the code at a specific time, e.g., 5:40 AM
    
##    target_time = "7:45"
##    while True:
##        current_time = time.strftime("%H:%M")
##        if current_time == target_time:
##            print("its time!")
##            analyze_all_coinbase_cryptos()
##            break
##        time.sleep(30)  # Check every 30 seconds
    
    
    # Run the analysis immediately
    analyze_all_coinbase_cryptos()
