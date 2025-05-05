README.txt

PROJECT DESCRIPTION

This is a Streamlit web application that predicts Tesla (TSLA) stock prices using a combination of news sentiment and historical stock data. It integrates - 

1. News Relevance Classification: Fetches recent Tesla news and filters for relevance.
2. Sentiment Analysis: Analyzes the tone of relevant news articles to quantify sentiment.
3. LSTM Price Prediction: Uses the sentiment score and 60 days of historical closing prices to predict the next day's stock price.
4. LLM Financial Advice: Offers reasoning for investment recommendations (BUY, HOLD, SELL) using Llama3 from Ollama.


REQUIREMENTS

- streamlit
- pandas
- numpy
- yfinance
- joblib
- plotly
- langchain
- langchain-community
- ollama (running locally with `llama3`)
- internet connection (for fetching stock data and news)
- scikit-learn
- torch
- tensorflow
- transformers

RUNNING THE APPLICATION

Ensure Ollama is installed and running with the `llama3` model:**
You can install Ollama from: https://ollama.com/


Launch the application with Streamlit:

streamlit run stockadvisor.py

HOW TO USE

Launch the application using the command above
Select a prediction date using the date picker


Click "Predict Stock Price" to start the analysis

Review the results:

Sentiment Score: Shows the overall sentiment of Tesla news
Predicted Price: The expected closing price for the next trading day
Price Chart: A visualization of historical prices and prediction
Suggested Action: BUY, SELL, or HOLD recommendation
LLM Financial Reasoning: Detailed explanation of the recommendation
Explore News Data: Click "Show News and Relevance" to see the news articles used for the analysis if you want.

UNDERSTANDING THE RESULTS

Sentiment Score

Positive values (above 0) indicate favorable news sentiment
Negative values (below 0) indicate unfavorable news sentiment
Values near 0 indicate neutral news sentiment

Suggested Actions

BUY: Predicted price increase of more than 2%
SELL: Predicted price decrease of more than 2%
HOLD: Predicted price change within ±2%

LLM Financial Reasoning
The application uses a large language model to provide context and interpretation for the prediction. This analysis considers:

Historical price trends
Price prediction
News sentiment
Market context


ACKNOWLEDGEMENTS

- ChatGPT (OpenAI)
- Claude AI (Anthropic)
- Langchain & Ollama developers
