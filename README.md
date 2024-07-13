Stock Price Analysis and Prediction Using Python
This project focuses on analyzing and predicting stock prices for the top 10 companies listed on the S&P 500 index using data from Yahoo Finance. The analysis covers data from January 1, 2012, to December 31, 2022. The project involves fetching data, performing exploratory data analysis (EDA), and making predictions using a Linear Regression model. The final goal is to forecast future stock prices.
Table of Contents
Installation
Data Fetching
Exploratory Data Analysis
Moving Average Calculation
Machine Learning
Stock Price Prediction
Evaluation
Documentation
Contributing
License
Installation
To run this project, you need to have Python and the necessary libraries installed. You can install the required libraries using the following command:

bash
Copy code
pip install requests pandas numpy yfinance matplotlib seaborn scikit-learn
Data Fetching
We start by fetching the top 10 companies by market capitalization from the S&P 500 index using Yahoo Finance and Wikipedia.

python
Copy code
import requests
import pandas as pd
import yfinance as yf

def fetch_sp500_companies():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = requests.get(url).content
    df_list = pd.read_html(html)
    sp500_df = df_list[0]
    return sp500_df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry', 'Headquarters Location']]

def fetch_market_caps(symbols):
    market_caps = {}
    for symbol in symbols:
        try:
            info = yf.Ticker(symbol).info
            if 'marketCap' in info:
                market_caps[symbol] = {'Market Cap': info['marketCap'], 'Company Name': info['longName'], 'Location': info['city']}
        except Exception as e:
            print(f"Error fetching market cap data for {symbol}: {e}")
            continue
    return market_caps

def fetch_top_stocks():
    sp500_symbols = fetch_sp500_companies()['Symbol'].tolist()
    market_caps = fetch_market_caps(sp500_symbols)
    top_stocks = sorted(market_caps.items(), key=lambda item: item[1]['Market Cap'], reverse=True)[:10]
    return top_stocks

# Fetch top stocks
top_stocks = fetch_top_stocks()

# Convert the result into a DataFrame
data = [{'Symbol': stock[0],
         'Company Name': stock[1]['Company Name'],
         'Location': stock[1]['Location'],
         'Market Cap': format_market_cap(stock[1]['Market Cap'])} for stock in top_stocks]

df = pd.DataFrame(data)

# Download stock data for all stocks available in df
stock_data = {}
for symbol in df['Symbol']:
    try:
        stock_data[symbol] = yf.download(symbol, start="2012-01-01", end="2022-12-31")
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        continue

# Convert stock_data into a single DataFrame
all_stock_data = pd.concat(stock_data.values(), keys=stock_data.keys(), names=['Symbol', 'Date'])
all_stock_data.reset_index(inplace=True)
all_stock_data = pd.merge(all_stock_data, df[['Symbol', 'Company Name', 'Location']], on='Symbol', how='left')
Exploratory Data Analysis
We perform EDA to understand the stock price trends and volatility.

python
Copy code
import matplotlib.pyplot as plt
import seaborn as sns

# Plot stock prices over time
plt.figure(figsize=(14, 8))
for symbol in df['Symbol']:
    stock_data_symbol = all_stock_data[all_stock_data['Symbol'] == symbol]
    plt.plot(stock_data_symbol['Date'], stock_data_symbol['Close'], label=symbol)
plt.title('Stock Prices Over Time for Top Stocks')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()

# Plot closing prices and volumes
num_stocks = len(df)
num_cols = 3
num_rows = (num_stocks + num_cols - 1) // num_cols

plt.figure(figsize=(15, 15))
plt.subplots_adjust(top=1.1, hspace=0.4)
for i, symbol in enumerate(df['Symbol'], 1):
    plt.subplot(num_rows, num_cols, i)
    stock_data_symbol = all_stock_data[all_stock_data['Symbol'] == symbol]
    plt.plot(stock_data_symbol['Date'], stock_data_symbol['Adj Close'])
    plt.ylabel('Adj Close')
    plt.xlabel('Date')
    plt.title(f"Closing Price of {df[df['Symbol'] == symbol]['Company Name'].iloc[0]}")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
Moving Average Calculation
We calculate the Linear Symmetric Moving Average (LSMA) to filter noise and observe price trends.

python
Copy code
import numpy as np

ma_periods = [84, 252, 1008, 2520]
for ma in ma_periods:
    column_name = f"LSMA for {ma} days"
    new_stock_data[column_name] = np.nan
    for i in range(ma-1, len(new_stock_data)):
        window_data = new_stock_data['Adj Close'].iloc[i-ma+1:i+1]
        x = np.arange(1, ma+1)
        slope, intercept = np.polyfit(x, window_data, 1)
        new_stock_data.at[new_stock_data.index[i], column_name] = slope * ma + intercept
Machine Learning
We use Linear Regression to model the stock prices.

python
Copy code
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to create X and y datasets
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

# Create X and y datasets for training
time_step = 60
X_train, y_train = create_dataset(dataset[:training_data_len], time_step)
X_test, y_test = create_dataset(dataset[training_data_len:], time_step)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
Stock Price Prediction
We predict future stock prices and visualize the results.

python
Copy code
# Create a DataFrame for actual and predicted prices
dates = df.index[training_data_len + time_step:].to_list()
predicted_df = pd.DataFrame({'Date': dates, 'Actual Price': y_test, 'Predicted Price': predictions})

# Plotting actual vs predicted
plt.figure(figsize=(14, 7))
plt.plot(dates, y_test, label='Actual Price', marker='o')
plt.plot(dates, predictions, label='Predicted Price', marker='x')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
Evaluation
We evaluate the model using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

python
Copy code
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'Root Mean Squared Error (RMSE): {rmse}')

mae = np.mean(np.abs(predictions - y_test))
print(f'Mean Absolute Error (MAE): {mae}')

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
rmse_cv = np.sqrt(-scores.mean())
print(f'Cross-Validated RMSE: {rmse_cv}')

residuals = y_test - predictions
plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=20)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
