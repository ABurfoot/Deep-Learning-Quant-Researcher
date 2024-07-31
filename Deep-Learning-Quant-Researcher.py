import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Converts the stock symbol to the format used by yfinance for different markets
def get_yfinance_symbol(symbol, market):
    market_extensions = {
        'LON': '.L',    # London Stock Exchange
        'ASX': '.AX',   # Australian Securities Exchange
        'TSE': '.T',    # Tokyo Stock Exchange
        'HKG': '.HK',   # Hong Kong Stock Exchange
        'SHA': '.SS',   # Shanghai Stock Exchange
        'SHE': '.SZ',   # Shenzhen Stock Exchange
        'FRA': '.F',    # Frankfurt Stock Exchange
        'ETR': '.DE',   # Deutsche Börse (XETRA)
        'BIT': '.MI',   # Borsa Italiana (Milan)
        'EPA': '.PA',   # Euronext Paris
        'AMS': '.AS',   # Euronext Amsterdam
        'EBR': '.BR',   # Euronext Brussels
        'ELI': '.LS',   # Euronext Lisbon
        'WSE': '.WA',   # Warsaw Stock Exchange
        'STO': '.ST',   # Stockholm Stock Exchange (NASDAQ OMX Nordic)
        'HEL': '.HE',   # Helsinki Stock Exchange (NASDAQ OMX Nordic)
        'CPH': '.CO',   # Copenhagen Stock Exchange (NASDAQ OMX Nordic)
        'ICE': '.IC',   # Iceland Stock Exchange (NASDAQ OMX Nordic)
        'TAL': '.TL',   # Tallinn Stock Exchange (NASDAQ OMX Baltic)
        'RIG': '.RG',   # Riga Stock Exchange (NASDAQ OMX Baltic)
        'VSE': '.VS',   # Vilnius Stock Exchange (NASDAQ OMX Baltic)
        'BME': '.MC',   # Bolsa de Madrid
        'SWX': '.SW',   # SIX Swiss Exchange
        'TSX': '.TO',   # Toronto Stock Exchange
        'TSXV': '.V',   # TSX Venture Exchange
        'BVC': '.CN',   # Bolsa de Valores de Colombia
        'BCBA': '.BA',  # Buenos Aires Stock Exchange
        'BVMF': '.SA',  # B3 (Brazil Bolsa Balcão)
        'NSE': '.NS',   # National Stock Exchange of India
        'BSE': '.BO',   # Bombay Stock Exchange
        'SGX': '.SI',   # Singapore Exchange
        'JSE': '.JO',   # Johannesburg Stock Exchange
        'TASE': '.TA',  # Tel Aviv Stock Exchange
        'NZX': '.NZ',   # New Zealand Exchange
    }
    return f"{symbol}{market_extensions.get(market, '')}"

# Returns the currency symbol for the given market
def get_currency_symbol(market):
    currency_map = {
        'NASDAQ': '$', 'NYSE': '$', 'AMEX': '$',  # United States
        'LON': '£',    # United Kingdom
        'ASX': 'A$',   # Australia
        'TSE': '¥',    # Japan
        'HKG': 'HK$',  # Hong Kong
        'SHA': '¥', 'SHE': '¥',  # China
        'FRA': '€', 'ETR': '€',  # Germany
        'BIT': '€',    # Italy
        'EPA': '€',    # France
        'AMS': '€',    # Netherlands
        'EBR': '€',    # Belgium
        'ELI': '€',    # Portugal
        'WSE': 'zł',   # Poland
        'STO': 'kr',   # Sweden
        'HEL': '€',    # Finland
        'CPH': 'kr',   # Denmark
        'ICE': 'kr',   # Iceland
        'TAL': '€',    # Estonia
        'RIG': '€',    # Latvia
        'VSE': '€',    # Lithuania
        'BME': '€',    # Spain
        'SWX': 'CHF',  # Switzerland
        'TSX': 'C$', 'TSXV': 'C$',  # Canada
        'BVC': 'COL$', # Colombia
        'BCBA': 'ARS$', # Argentina
        'BVMF': 'R$',  # Brazil
        'NSE': '₹', 'BSE': '₹',  # India
        'SGX': 'S$',   # Singapore
        'JSE': 'R',    # South Africa
        'TASE': '₪',   # Israel
        'NZX': 'NZ$',  # New Zealand
    }
    return currency_map.get(market, '$')  # Default to $ if market not found

# Returns the market index symbol for the given market
def get_market_index(market):
    market_indices = {
        'NYSE': '^GSPC',    # S&P 500
        'NASDAQ': '^IXIC',  # NASDAQ Composite
        'LON': '^FTSE',     # FTSE 100 (UK)
        'TSE': '^N225',     # Nikkei 225 (Japan)
        'HKG': '^HSI',      # Hang Seng Index (Hong Kong)
        'SHA': '000001.SS', # Shanghai Composite Index
        'SHE': '399001.SZ', # Shenzhen Component Index
        'FRA': '^GDAXI',    # DAX (Germany)
        'PAR': '^FCHI',     # CAC 40 (France)
        'ASX': '^AXJO',     # S&P/ASX 200 (Australia)
        'TSX': '^GSPTSE',   # S&P/TSX Composite Index (Canada)
        'NSE': '^NSEI',     # NIFTY 50 (India)
        'BSE': '^BSESN',    # S&P BSE SENSEX (India)
        'SGX': '^STI',      # Straits Times Index (Singapore)
        'JSE': '^J203',     # FTSE/JSE All Share Index (South Africa)
        'NZX': '^NZ50',     # S&P/NZX 50 Index (New Zealand)
    }
    return market_indices.get(market, '^GSPC')  # Default to S&P 500 if not found

# Calculates the Relative Strength Index (RSI) for the given price series
def calculate_rsi(prices, window=14):
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100./(1. + rs)

    for i in range(window, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(window - 1) + upval)/window
        down = (down*(window - 1) + downval)/window
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return rsi

# Calculates the Bollinger Bands for a given price series
def calculate_bollinger_bands(close, window=20, num_std=2):
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

# Calculates the Average True Range (ATR) for a given price series
def calculate_atr(high, low, close, period=14):
    tr = np.maximum(high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1)))
    atr = tr.rolling(window=period).mean()
    return atr

# Calculates the On-Balance Volume (OBV) indicator
def calculate_obv(close, volume):
    obv = (np.sign(close.diff()) * volume).cumsum()
    return obv

# Calculates the Average Directional Index (ADX) indicator
def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(period).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/period).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (period - 1)) + dx) / period
    adx_smooth = adx.ewm(alpha=1/period).mean()
    return adx_smooth

# Calculates the Stochastic Oscillator for a given price series
def calculate_stochastic_oscillator(high, low, close, k_window=14):
    low_min = low.rolling(k_window).min()
    high_max = high.rolling(k_window).max()
    
    k = 100 * ((close - low_min) / (high_max - low_min))
    return k

# Checks the quality of the input data and prints warnings for potential issues
def check_data_quality(df):
    missing_pct = df.isnull().mean() * 100
    if missing_pct.max() > 5:
        print(f"Warning: High percentage of missing data in some columns: {missing_pct[missing_pct > 5]}")
    
    stale_data = (df['Close'] == df['Close'].shift()).sum() / len(df) * 100
    if stale_data > 10:
        print(f"Warning: {stale_data:.2f}% of closing prices are unchanged. Possible stale data.")
    
    volume_zero = (df['Volume'] == 0).sum() / len(df) * 100
    if volume_zero > 5:
        print(f"Warning: {volume_zero:.2f}% of trading volumes are zero. Possible data quality issue.")

# Prepares the input data for the model, including feature engineering and scaling
def prepare_data(symbol, market, start_date, end_date, sequence_length):
    yf_symbol = get_yfinance_symbol(symbol, market)
    
    data = yf.download(yf_symbol, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data available for {symbol} on {market} in the specified date range.")
    
    data = data[data.index.dayofweek < 5]
    
    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    
    df['Bollinger_Upper'], df['Bollinger_Lower'] = calculate_bollinger_bands(df['Close'])
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
    df['OBV'] = calculate_obv(df['Close'], df['Volume'])
    df['ADX'] = calculate_adx(df['High'], df['Low'], df['Close'])
    df['Stochastic_K'] = calculate_stochastic_oscillator(df['High'], df['Low'], df['Close'])
    
    df['Price_Momentum'] = df['Close'].pct_change(5)
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    market_index = get_market_index(market)
    index_data = yf.download(market_index, start=start_date, end=end_date)['Close']
    df['MarketIndex'] = index_data
    df['Market_Correlation'] = df['Close'].rolling(window=30).corr(index_data)
    
    df['DayOfWeek'] = df.index.dayofweek
    df['MonthOfYear'] = df.index.month
    df['QuarterOfYear'] = df.index.quarter
    
    df = df.dropna()
    
    check_data_quality(df)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    scaler.feature_names_in_ = df.columns.tolist()
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length, df.columns.get_loc('Close')])
    
    X, y = np.array(X), np.array(y)
    
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).unsqueeze(1)
    
    return X, y, scaler, data, df

# Defines the LSTM-based neural network model for stock price prediction
class ImprovedStockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(ImprovedStockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc1(lstm_out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Trains the stock prediction model
def train_model(model, X_train, y_train, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        
        if torch.isnan(loss):
            print(f"NaN loss encountered at epoch {epoch+1}. Stopping training.")
            return None
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

# Evaluates the trained model on test data and calculates performance metrics
def evaluate_model(model, X_test, y_test, scaler):
    model.eval()
    with torch.no_grad():
        if X_test.shape[0] == 0:
            print("Warning: Empty test set. Cannot evaluate.")
            return None, None
        
        y_pred = model(X_test)
        
        y_pred_dummy = pd.DataFrame(np.zeros((len(y_pred), len(scaler.feature_names_in_))), 
                                    columns=scaler.feature_names_in_)
        y_pred_dummy['Close'] = y_pred.numpy().flatten()
        
        y_test_dummy = pd.DataFrame(np.zeros((len(y_test), len(scaler.feature_names_in_))), 
                                    columns=scaler.feature_names_in_)
        y_test_dummy['Close'] = y_test.numpy().flatten()
        
        y_pred = scaler.inverse_transform(y_pred_dummy)[:, list(scaler.feature_names_in_).index('Close')]
        y_test = scaler.inverse_transform(y_test_dummy)[:, list(scaler.feature_names_in_).index('Close')]
        
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        print(f'Mean Absolute Error: {mae:.4f}')
        print(f'Mean Absolute Percentage Error: {mape:.4f}')
    return y_pred, y_test

# Predicts stock prices for the next n days using the trained model
def predict_next_n_days(model, last_sequence, scaler, n_days, df):
    model.eval()
    predictions = []
    current_sequence = last_sequence.clone()
    
    ma_window = min(30, len(df))
    last_ma = df['Close'].iloc[-ma_window:].mean()
    
    last_known_values = pd.DataFrame(scaler.inverse_transform(current_sequence[-1].unsqueeze(0)), 
                                     columns=scaler.feature_names_in_)
    
    recent_volatility = df['Close'].pct_change().rolling(window=20).std().iloc[-1]
    
    for _ in range(n_days):
        with torch.no_grad():
            model_output = model(current_sequence.unsqueeze(0))
            
            price_change = model_output.item()
            
            last_price = last_known_values['Close'].iloc[0]
            
            lstm_weight = 0.7
            ma_weight = 0.3
            predicted_price = (last_price + price_change) * lstm_weight + last_ma * ma_weight
            
            if predictions:
                predicted_price = 0.7 * predicted_price + 0.3 * predictions[-1]
            
            volatility_factor = np.random.normal(1, recent_volatility)
            predicted_price *= volatility_factor
            
            predictions.append(predicted_price)
            
            last_known_values.loc[0, 'Close'] = predicted_price
            
            last_known_values.loc[0, 'DayOfWeek'] = (last_known_values.loc[0, 'DayOfWeek'] + 1) % 5
            last_known_values.loc[0, 'Volatility'] = recent_volatility
            
            new_row = scaler.transform(last_known_values)
            
            current_sequence = torch.cat((current_sequence[1:], torch.FloatTensor(new_row).squeeze(0).unsqueeze(0)), 0)
            
            last_ma = 0.9 * last_ma + 0.1 * predicted_price
    
    return np.array(predictions)

# Performs walkforward analysis to train and evaluate the model on different data subsets
def walkforward_analysis(X, y, scaler, df, sequence_length, hidden_dim, num_layers, epochs, lr):
    best_model = None
    best_mae = float('inf')
    n_splits = 5

    for i in range(n_splits):
        train_end = int((i + 1) * len(X) / n_splits)
        
        X_train, X_val = X[:train_end], X[train_end:]
        y_train, y_val = y[:train_end], y[train_end:]

        print(f"Fold {i+1} - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"Fold {i+1} - X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

        model = ImprovedStockPredictor(X.shape[2], hidden_dim, num_layers, 1)
        trained_model = train_model(model, X_train, y_train, epochs, lr)

        if trained_model is None:
            print(f"Model training failed for fold {i+1}. Skipping this fold.")
            continue

        if i == n_splits - 1:
            y_pred, y_true = evaluate_model(trained_model, X_train, y_train, scaler)
        else:
            y_pred, y_true = evaluate_model(trained_model, X_val, y_val, scaler)
        
        if y_pred is None or y_true is None or len(y_pred) == 0 or len(y_true) == 0:
            print(f"Evaluation failed for fold {i+1}. Skipping this fold.")
            continue

        mae = mean_absolute_error(y_true, y_pred)
        print(f"Fold {i+1} MAE: {mae:.4f}")

        if mae < best_mae:
            best_mae = mae
            best_model = trained_model.state_dict()

    if best_model is None:
        raise ValueError("Walkforward analysis failed. No valid model was produced.")

    print(f"Best MAE: {best_mae:.4f}")
    return best_model, best_mae

# Plots the results of the walkforward analysis, comparing actual and predicted prices
def plot_walkforward_analysis(dates, actual, predicted, symbol, market, currency_symbol):
    plt.figure(figsize=(20,10))
    plt.plot(dates, actual[:len(predicted)], label='Actual', linewidth=2)
    plt.plot(dates, predicted, label='Predicted', linewidth=2)
    plt.legend(fontsize=12)
    plt.title(f"{symbol} Stock Price Prediction (Walkforward Analysis)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(f"Price ({currency_symbol})", fontsize=12)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=10)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{currency_symbol}{x:,.2f}'))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# Plots the predicted future stock prices
def plot_future_predictions(future_dates, next_n_days_prediction, symbol, market, prediction_days, currency_symbol):
    plt.figure(figsize=(20,10))
    plt.plot(future_dates, next_n_days_prediction, label='Predicted', linewidth=2, marker='o', markersize=6)
    plt.legend(fontsize=12)
    plt.title(f"{symbol} Stock Price Prediction (Next {prediction_days} Business Days)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(f"Predicted Price ({currency_symbol})", fontsize=12)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(20))
    plt.xticks(rotation=90, ha='center', fontsize=8)
    plt.yticks(fontsize=10)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{currency_symbol}{x:,.2f}'))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# Main execution block
if __name__ == "__main__":
    symbol = input("Enter the stock symbol (e.g., AAL for American Airlines Group Inc on NASDAQ): ").upper()
    market = input("Enter the market (e.g., NASDAQ, LON, ASX): ").upper()
    start_date = "2022-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    sequence_length = 60
    hidden_dim = 128
    num_layers = 3
    epochs = 100
    lr = 0.001
    prediction_days = int(input("Enter the number of days you want to predict: "))

    currency_symbol = get_currency_symbol(market)

    X, y, scaler, data, df = prepare_data(symbol, market, start_date, end_date, sequence_length)

    best_model_state, best_mae = walkforward_analysis(X, y, scaler, df, sequence_length, hidden_dim, num_layers, epochs, lr)

    model = ImprovedStockPredictor(X.shape[2], hidden_dim, num_layers, 1)
    model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        y_pred_dummy = pd.DataFrame(np.zeros((len(y_pred), len(df.columns))), columns=df.columns)
        y_pred_dummy['Close'] = y_pred.numpy().flatten()
        y_pred = scaler.inverse_transform(y_pred_dummy)[:, df.columns.get_loc('Close')]

    actual = df['Close'][sequence_length:].values
    predicted = y_pred
    dates = data.index[sequence_length:sequence_length+len(predicted)]
    plot_walkforward_analysis(dates, actual, predicted, symbol, market, currency_symbol)

    last_sequence = torch.FloatTensor(scaler.transform(df.iloc[-sequence_length:]))
    next_n_days_prediction = predict_next_n_days(model, last_sequence, scaler, prediction_days, df)

    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days, freq='B')

    plot_future_predictions(future_dates, next_n_days_prediction, symbol, market, prediction_days, currency_symbol)

    print(f"\nPredicted prices for {symbol} ({market}) over the next {prediction_days} business days:")
    for date, price in zip(future_dates, next_n_days_prediction):
        print(f"{date.strftime('%Y-%m-%d')}: {currency_symbol}{price:.2f}")