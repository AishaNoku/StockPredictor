import joblib
import pandas as pd
import torch
import torch.nn.functional as F


class SentimentAnalysis:
    def __init__(self):
        self.model = joblib.load("finbert_model.joblib")
        self.tokenizer = joblib.load("finbert_tokenizer.joblib")
    
    def get_sentiment_probabilities(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
        return probs  # [positive, neutral, negative]
    
    def analyze_dataframe(self, df):
        # Get sentiment probabilities for each article
        sentiments = df['title'].apply(self.get_sentiment_probabilities)
        df[['Positive', 'Neutral', 'Negative']] = pd.DataFrame(sentiments.tolist(), index=df.index)
        avg_sentiment = df[['Positive', 'Neutral', 'Negative']].mean()
        sentiment_score = avg_sentiment['Positive'] - avg_sentiment['Negative']
        return sentiment_score  
