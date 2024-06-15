import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle
import matplotlib.pyplot as plt
import base64

# Load the pre-trained model (assuming it's saved as a pickle file)
with open('final_model.pkl', 'rb') as f:
    final_model = pickle.load(f)

# Load historical data from CSV
historical_data = pd.read_csv('gold_monthly_csv.csv', parse_dates=['Date'], index_col='Date')

# Ensure the index is of datetime type
historical_data.index = pd.to_datetime(historical_data.index)

# Define the prediction function
def predict_gold_rate(date):
    # Generate dates from the last known date to the input date
    last_date = historical_data.index[-1]
    future_dates = pd.date_range(start=last_date, end=date, freq='D')
    
    # Forecast the values for the future dates
    forecast = final_model.forecast(len(future_dates))
    
    return forecast[-1], future_dates, forecast  # Return the forecasted value for the input date and forecast series

# HTML and CSS styling
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gold Rate Predictor</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f9;
            color: #333;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #4CAF50;
        }
        p {
            text-align: center;
            font-size: 1.2em;
        }
        .prediction {
            text-align: center;
            font-size: 1.5em;
            margin-top: 20px;
        }
        .plot-container {
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Gold Rate Predictor</h1>
        <p>This app predicts the gold rate for a given date using an Exponential Smoothing model.</p>
    </div>
</body>
</html>
"""

# Function to render HTML in Streamlit
def render_html(html_code):
    b64_html = base64.b64encode(html_code.encode()).decode()
    st.markdown(f'<iframe src="data:text/html;base64,{b64_html}" width="100%" height="300"></iframe>', unsafe_allow_html=True)

# Create the Streamlit app
def main():
    # Display the HTML content
    render_html(html_content)
    
    # Input date from the user
    input_date = st.date_input("Enter a date to predict the gold rate:")

    if input_date:
        prediction, future_dates, forecast = predict_gold_rate(input_date)
        
        st.write(f"The predicted gold rate for {input_date} is {prediction:.2f}")
        
        st.write("Forecasted Gold Rates:")
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(historical_data.index, historical_data['Price'], label='Actual')
        ax.plot(future_dates, forecast, label='Predicted', alpha=0.7)
        ax.fill_between(future_dates, forecast - 2*forecast.std(), forecast + 2*forecast.std(), color='k', alpha=0.1)
        ax.set_title('Gold Rate Prediction')
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
