"# Deep-Learning-Quant-Researcher" 

## Overview
This project is a sophisticated stock price prediction tool that uses deep learning techniques to forecast future stock prices. It employs an LSTM (Long Short-Term Memory) neural network model and incorporates various technical indicators for enhanced prediction accuracy.

## Features
- Supports multiple stock markets and currencies worldwide
- Utilizes yfinance for real-time stock data retrieval
- Implements walk-forward analysis for robust model evaluation
- Incorporates multiple technical indicators (RSI, Bollinger Bands, ATR, OBV, ADX, Stochastic Oscillator)
- Provides data quality checks to ensure reliable input
- Generates visualizations for both historical analysis and future predictions

## Requirements
- Python 3.7+
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib
- yfinance

## Installation
1. Clone this repository: https://github.com/ABurfoot/Deep-Learning-Quant-Researcher.git
2. Install the required packages:
  pip install yfinance
  pip install pandas
  pip install numpy
  pip install torch
  pip install scikit-learn
  pip install matplotlib

You will be prompted to enter:
1. The stock symbol (e.g., AAPL for Apple Inc.)
2. The market (e.g., NASDAQ, LON, ASX)
3. The number of days you want to predict

The program will then:
1. Retrieve historical data
2. Perform walk-forward analysis to train and evaluate the model
3. Display graphs of the walk-forward analysis results
4. Predict stock prices for the specified number of future days
5. Display a graph of the future predictions

## How it Works
1. **Data Retrieval**: Uses yfinance to download historical stock data.
2. **Data Preparation**: Calculates technical indicators (RSI, Bollinger Bands, ATR, OBV, ADX, Stochastic Oscillator) and prepares sequences for the LSTM model.
3. **Model Training**: Employs an LSTM neural network, trained using walk-forward analysis.
4. **Evaluation**: Uses Mean Absolute Error (MAE) to assess model performance.
5. **Prediction**: Forecasts future stock prices based on the trained model and recent data.

## Key Components
- `get_yfinance_symbol()`: Converts stock symbols for different markets.
- `prepare_data()`: Prepares and engineers features from raw stock data.
- `ImprovedStockPredictor`: LSTM-based neural network model.
- `walkforward_analysis()`: Implements the walk-forward validation technique.
- `predict_next_n_days()`: Generates future stock price predictions.

## Limitations
- The stock market is inherently unpredictable and affected by many external factors. This tool should be used for educational purposes only and not for actual trading decisions.
- The accuracy of predictions can vary significantly depending on the stock and market conditions.

## Disclaimer
This software is for educational and research purposes only. Do not use it to make any type of investment decisions. Always consult with a qualified financial advisor before making investment decisions. The creator (ABurfoot) of this project is not responsible for any financial losses incurred from using this tool.
