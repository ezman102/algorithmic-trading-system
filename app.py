import streamlit as st
import pandas as pd
import joblib
from datetime import date
from main import fetch_data, add_technical_indicators, preprocess_data

# Assuming your models are saved in the specified paths
model_classification_path = 'C:/Users/User/Desktop/algorithmic-trading-system/best_random_forest_model_accuracy_0.6552.joblib'
model_regression_path = 'C:/Users/User/Desktop/algorithmic-trading-system/best_random_forest_regression_model.joblib'
model_classification = joblib.load(model_classification_path)
model_regression = joblib.load(model_regression_path)

# Streamlit app
st.title('Stock Prediction App')

# Sidebar user input
mode = st.sidebar.radio("Select Mode:", ('Classification', 'Regression'))

stock_symbol = st.sidebar.text_input('Stock Symbol', value='NVDA')
start_date = st.sidebar.date_input('Start Date', date(2023, 1, 1))
end_date = st.sidebar.date_input('End Date', date(2024, 2, 28))

if st.sidebar.button('Predict'):
    data = fetch_data(stock_symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    if data is not None and not data.empty:
        data_processed = add_technical_indicators(data, [
        # {'name': 'SMA_10', 'type': 'SMA', 'window': 10},
        # {'name': 'SMA_30', 'type': 'SMA', 'window': 30},
        # {'name': 'EMA_10', 'type': 'EMA', 'window': 10},
        # {'name': 'EMA_30', 'type': 'EMA', 'window': 30},
        # {'name': 'RSI_14', 'type': 'RSI', 'window': 14},
        # {'name': 'MACD', 'type': 'MACD', 'short_window': 12, 'long_window': 26, 'signal_window': 9},
        # {'name': 'MACD_Signal', 'type': 'MACD', 'short_window': 12, 'long_window': 26, 'signal_window': 9},
        {'name': 'MACD_Histogram', 'type': 'MACD', 'short_window': 12, 'long_window': 26, 'signal_window': 9},
        # {'name': 'BB_Upper', 'type': 'BB', 'window': 20}, 
        # {'name': 'BB_Lower', 'type': 'BB', 'window': 20}, 
        {'name': 'ATR', 'type': 'ATR', 'window': 14},  # Average True Range
        # {'name': 'Stochastic_Oscillator', 'type': 'Stochastic', 'window': 14},
        # {'name': 'OBV', 'type': 'OBV'}  
    ],drop_original=True)
        data_processed = preprocess_data(data_processed)
        
        if mode == 'Classification':
            print(data_processed.shape)
            prediction = model_classification.predict(data_processed)
            st.write('Classification Prediction:', prediction)
        elif mode == 'Regression':
            prediction = model_regression.predict(data_processed)
            st.write('Regression Prediction:', prediction)
            st.line_chart(prediction) 
    else:
        st.write("No data available for the given inputs.")
