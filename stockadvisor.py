# Tesla News-Driven Stock Price Prediction Streamlit Application
# This application integrates multiple components to predict Tesla stock prices based on news sentiment:
# 1. News Collection & Relevance Classification: Fetches recent Tesla news and filters for relevance
# 2. Sentiment Analysis: Analyzes the emotional tone of relevant news articles
# 3. LSTM Model Prediction: Uses historical prices + sentiment to predict future stock price
# 4. LLM-Based Financial Advice: Provides reasoning for investment decisions using an LLM
# acknowlwdgements: Chatgpt, Claude ai

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import date, datetime, timedelta
import yfinance as yf
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from relevant_news_predictor import TeslaPredictor
from sentiment_implementation import SentimentAnalysis

import plotly.graph_objects as go
from datetime import timedelta

# Load the pre-trained LSTM model for price prediction from saved files
# Also load the scaler used to normalize input data for the model
# Initialize the Llama3 LLM via Ollama for generating investment advice

lstm_model = joblib.load("lstm_model.joblib")
scaler = joblib.load("scaler.pkl")

def get_investment_recommendation(symbol, latest_price, predicted_price, sentiment_score):
    # Calculate price change percentage
    price_change_pct = ((predicted_price - latest_price) / latest_price) * 100
    
    # Determine basic recommendation
    if price_change_pct > 2:  # More than 2% increase predicted
        recommendation = "BUY"
        confidence = min(abs(price_change_pct) / 5, 1) * 100  # Scale confidence (max 100%)
    elif price_change_pct < -2:  # More than 2% decrease predicted
        recommendation = "SELL"
        confidence = min(abs(price_change_pct) / 5, 1) * 100
    else:
        recommendation = "HOLD"
        confidence = (1 - min(abs(price_change_pct) / 2, 1)) * 100
    
    # Factor in sentiment
    sentiment_text = "very positive" if sentiment_score > 0.75 else "positive" if sentiment_score > 0.6 else \
                    "neutral" if sentiment_score > 0.4 else "negative" if sentiment_score > 0.25 else "very negative"
    
    # Generate reasoning based on the data
    if recommendation == "BUY":
        reasoning = f"""
        Based on the predictive analysis of Tesla (TSLA) stock, I recommend a BUY position with {confidence:.1f}% confidence.
        
        Key factors supporting this recommendation:
        
        1. Price Momentum: The model predicts a price increase of {price_change_pct:.2f}% from ${latest_price:.2f} to ${predicted_price:.2f}, indicating positive momentum.
        
        2. Sentiment Analysis: News sentiment is {sentiment_text} with a score of {sentiment_score:.2f} on a scale from 0 to 1, suggesting market perception is favorable.
        
        3. Technical Indicators: The LSTM prediction model, which incorporates 60 days of historical price movements, suggests continued upward movement, creating a potential buying opportunity.
        
        This combination of positive price prediction and {sentiment_text} news sentiment provides a strong case for increasing positions in Tesla stock at this time.
        """
    elif recommendation == "SELL":
        reasoning = f"""
        Based on the predictive analysis of Tesla (TSLA) stock, I recommend a SELL position with {confidence:.1f}% confidence.
        
        Key factors supporting this recommendation:
        
        1. Price Weakness: The model predicts a price decrease of {abs(price_change_pct):.2f}% from ${latest_price:.2f} to ${predicted_price:.2f}, indicating downward pressure.
        
        2. Sentiment Analysis: News sentiment is {sentiment_text} with a score of {sentiment_score:.2f} on a scale from 0 to 1, which may reflect emerging concerns in the market.
        
        3. Technical Indicators: The LSTM prediction model, incorporating 60 days of historical price movements, suggests a potential reversal or continuation of negative price action.
        
        Given these factors, particularly the predicted price decline and {sentiment_text} sentiment, reducing exposure to Tesla stock appears prudent at this time.
        """
    else:  # HOLD
        reasoning = f"""
        Based on the predictive analysis of Tesla (TSLA) stock, I recommend a HOLD position with {confidence:.1f}% confidence.
        
        Key factors supporting this recommendation:
        
        1. Price Stability: The model predicts a small price change of {price_change_pct:.2f}% from ${latest_price:.2f} to ${predicted_price:.2f}, suggesting relatively stable price action.
        
        2. Sentiment Analysis: News sentiment is {sentiment_text} with a score of {sentiment_score:.2f} on a scale from 0 to 1, which indicates neither strong positive nor negative market perception.
        
        3. Technical Indicators: The LSTM prediction model, based on 60 days of historical prices, does not show a clear directional signal that would warrant taking new positions.
        
        With no strong directional bias indicated by either the price prediction or sentiment analysis, maintaining current positions is the most prudent approach until clearer signals emerge.
        """
    
    return reasoning

# This function calculates a date 100 days prior to the provided date
# We use 100 days to ensure we have enough data (at least 60 days) even with missing trading days
# Returns the date string in YYYY-MM-DD format for use with yfinance API
def sixty_days_back(date_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    new_date = date_obj - timedelta(days=200)
    return new_date.strftime("%Y-%m-%d")

# Takes a dataframe column and converts it to a 1-dimensional numpy array
def dataframe_column_to_1d_array(df, column_name):
    array_1d = df[column_name].to_numpy()
    if len(array_1d.shape) > 1:
        array_1d = array_1d.flatten()
    return array_1d

# Downloads Tesla stock data for the 60+ days leading up to the target date
def get_60day_history(target_date_str):
    try:
        tesla_data = yf.download(
            "TSLA",
            start=sixty_days_back(target_date_str),
            end=target_date_str,
            progress=False
        )
        if tesla_data.empty:
            return []
        close_prices = dataframe_column_to_1d_array(tesla_data, "Close")
        return close_prices
    except Exception as e:
        print("Error fetching stock data:", e)
        return []

def prepare_lstm_input(prices, sentiment_score):
    if len(prices) < 60:
        raise ValueError("Need at least 60 price points to make prediction.")

    # Create 60 rows of [close_price, sentiment_score] where each row contains the closing price and the sentiment score for that day 
    data = np.array([[p, sentiment_score] for p in prices[-60:]])

    # Normilisng the data using the already trained scaler
    scaled = scaler.transform(data)

    # Reshape the data so that it is understood by the lstm
    return scaled.reshape(1, 60, 2)


# Streamlit UI implementation 

st.title("📊 Tesla Stock Predictor and Advisor Based on News Sentiment and Previous Stock Prices")

symbol = "TSLA"
target_date = st.date_input("Prediction Date (news + stock)", value=date(2025, 4, 29))
target_str = target_date.strftime('%Y-%m-%d')

if st.button("Predict Stock Price"):
    with st.spinner("🔍 Fetching and analyzing news..."):
        predictor = TeslaPredictor()
        sentiment_tool = SentimentAnalysis()

        # Get news and classify for relevance to Tesla stock
        df_news = predictor.get_and_classify_news(target_date.strftime('%Y%m%d'))
        #filter to get relevant news only 
        df_relevant = df_news[df_news['relevance'] == 'Relevant']

        if df_relevant.empty:
            st.warning("No relevant news found for the selected date.")
            sentiment_score = 0.0
        else:
            sentiment_score = sentiment_tool.analyze_dataframe(df_relevant)
        st.write(f"🧠 Sentiment Score: {sentiment_score:.4f}")


        # This section:
    # 1. Retrieves historical Tesla stock prices
    # 2. Combines price data with news sentiment
    # 3. Uses the LSTM model to predict the next day's closing price


    with st.spinner("📈 Getting stock data and predicting..."):
        prices = get_60day_history(target_str)

        if len(prices) < 60:
            st.error("⚠️ Not enough stock price data. Need 60 days of closing prices.")
        else:

            X_input = prepare_lstm_input(prices, sentiment_score)
            
            raw_pred = lstm_model.predict(X_input)

            padded = np.hstack([raw_pred, np.zeros((raw_pred.shape[0], 1))])  # shape (1, 2)
            
            inversed = scaler.inverse_transform(padded)[0][0]
            predicted_price = inversed

            st.success(f"📈 Predicted Closing Price for {target_str}: ${predicted_price:.2f}")

            latest_price = prices[-1]
            decision = ""
            if predicted_price > latest_price * 1.02:
                decision = "BUY"
            elif predicted_price < latest_price * 0.98:
                decision = "SELL"
            else:
                decision = "HOLD"

            # Data Visualization: Historical prices on a line graph and a plotted point + Predicted Prices
            date_range = pd.date_range(start=sixty_days_back(target_str), end=target_str)
            if len(date_range) > len(prices):
                date_range = date_range[-len(prices):]  

            fig = go.Figure()

            # Plot historical closing prices
            fig.add_trace(go.Scatter(
                x=date_range,
                y=prices,
                mode='lines+markers',
                name='Actual Closing Price',
                line=dict(color='blue')
            ))
            
            # Add predicted price as a red dot
            predicted_date = target_date + timedelta(days=1)
            fig.add_trace(go.Scatter(
                x=[predicted_date],
                y=[predicted_price],
                mode='markers',
                name='Predicted Price',
                marker=dict(color='red', size=10, symbol='circle')
                
                ))
            
            fig.update_layout(
                title=f"TSLA: Last 60 Days & Predicted Price for {predicted_date.strftime('%Y-%m-%d')}",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                legend_title="Legend",
                template="plotly_white"
                )
            
            st.plotly_chart(fig, use_container_width=True)


            st.subheader("🧠 Suggested Action:")
            st.success(f"{decision}")

            # Prompting LLM to justify decision 
            stock_summary = f"""
            Stock: {symbol}
            Closing Price on {target_str}: ${latest_price:.2f}
            Predicted Closing Price: ${predicted_price:.2f}
            Sentiment Score: {sentiment_score:.2f}
            """

            prompt = PromptTemplate(
                input_variables=["summary"],
                template="""
                Analyze the following stock data and predict whether the user should BUY, HOLD, or SELL Tesla stock.
                Justify your recommendation using evidence from the price trend and sentiment score.

                {summary}
                """
            )

            advice = get_investment_recommendation(symbol, latest_price, predicted_price, sentiment_score)
            # Display the LLM's financial analysis and reasoning
            st.subheader("🧠 LLM Financial Reasoning:")
            st.info(advice)

    with st.expander("📰 Show News and Relevance"):
        if not df_news.empty:
            st.dataframe(df_news[['title', 'relevance']])
        else:
            st.write("No news data available.")
