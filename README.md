# Stock Analysis and Prediction Using Machine Learning

## Overview
This project involves analyzing and predicting stock prices using historical data from Yahoo Finance. The analysis covers the top 10 companies by market capitalization in the S&P 500 index. The project includes data fetching, exploratory data analysis (EDA), and predictive modeling using linear regression.

**Code:** [`Stock Price Prediction`]( https://github.com/MohdIllham/Stock-Prediction-Price/blob/main/Stock%20Price%20Prediction.ipynb)

## Table of Contents
- [Overview](#Overview)
- [Installation](#Installation)
- [Data Collection](#Data-Collection)
- [Data Preparation](#Data-Preparation)
- [Exploratory Data Analysis(EDA)](#Exploratory-Data-Analysis-EDA)
- [Predictive Modeling](#Predictive-Modeling)
- [Evaluation Metrics](#Evaluation-Metrics)
- [Future Work](#Future-Work)

## Installation
```
pip install requests pandas numpy yfinance matplotlib seaborn scikit-learn
```
## Data Collection
We fetch the list of S&P 500 companies from Wikipedia and their historical stock data from Yahoo Finance using the yfinance library. The top 10 companies are selected based on their market capitalization.
Steps:
We fetch the list of S&P 500 companies from Wikipedia and their historical stock data from Yahoo Finance using the yfinance library. The top 10 companies are selected based on their market capitalization.
Steps:
1.	Fetch the list of S&P 500 companies from Wikipedia.
2.	Fetch historical stock data for the top 10 companies from Yahoo Finance.
3.	Merge the stock data with company information.

## Data Preparation
Data preparation involves:
1.	Cleaning the data: Handling missing values, formatting dates, etc.
2.	Calculating additional metrics: Daily returns, yearly returns, and least squares moving averages (LSMA).
3.	Preparing the data for machine learning models: Splitting into training and testing sets.

## Exploratory Data Analysis (EDA)
In this section, we perform various analyses to understand the historical performance of the selected stocks:
•	Plotting stock prices over time.
•	Analyzing closing prices to identify trends and anomalies.
•	Evaluating trading volumes to assess market activity.
•	Calculating and plotting yearly returns to understand long-term performance.
•	Using Kernel Density Estimation (KDE) plots to visualize the distribution of yearly returns.
•	Applying Least Squares Moving Average (LSMA) to smooth out price movements and reduce noise.

## Predictive Modeling
For the stock price prediction, we:
•	Select Apple Inc. (AAPL) as the target stock for prediction.
•	Use a linear regression model from scikit-learn.
•	Split the historical data into training and testing sets.
•	Train the model on the training data and generate predictions on the testing data.
•	Forecast future prices for a specified period (e.g., 30 days).

## Evaluation Metrics
We evaluate the model's performance using:
•	Root Mean Squared Error (RMSE)
•	Mean Absolute Error (MAE)
•	Cross-validated RMSE to ensure the robustness of the model.
•	Plotting the residuals to check for any patterns or biases.

## Future Work
Potential improvements and future work include:
•	Using more complex models (e.g., LSTM, ARIMA) for better accuracy.
•	Incorporating additional features such as economic indicators or sentiment analysis from news articles.
•	Extending the analysis to a broader range of stocks or indices.
•	Implementing a real-time stock prediction system with live data updates.



