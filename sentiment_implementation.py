import joblib
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

class SentimentAnalysis:
    def __init__(self):
        # Load model and tokenizer directly from Hugging Face
        self.model_name = "yiyanghkust/finbert-tone"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()  # Set to evaluation mode
    
    def get_sentiment_probabilities(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
        
        # Make sure we're returning a list even for single examples
        if not isinstance(probs[0], list) and len(probs) == 3:
            return probs  # [positive, neutral, negative] or [negative, neutral, positive]
        return probs
    
    def analyze_dataframe(self, df):
        # Get sentiment probabilities for each article
        sentiments = df['title'].apply(self.get_sentiment_probabilities)
        
        # Map the outputs to the expected format
        # Note: Check the label order of the model - may need adjustment
        df[['Negative', 'Neutral', 'Positive']] = pd.DataFrame(sentiments.tolist(), index=df.index)
        
        avg_sentiment = df[['Positive', 'Neutral', 'Negative']].mean()
        sentiment_score = avg_sentiment['Positive'] - avg_sentiment['Negative']
        return sentiment_score
