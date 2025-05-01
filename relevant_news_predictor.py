
import joblib
import re
import os
import pandas as pd

class TeslaPredictor:
    def __init__(self):
        # Load model and vectorizer
        self.model = joblib.load('tesla_model.joblib')
        self.vectorizer = joblib.load('tfidf_vectorizer.joblib')
    
    def clean_text(self, text):
        text = str(text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        return re.sub('\s+', ' ', text).lower().strip()

    def predict(self, text):
        cleaned = self.clean_text(text)
        vector = self.vectorizer.transform([cleaned])
        return "Relevant" if self.model.predict(vector)[0] == 1 else "Irrelevant"

    def get_from_internet(self, date):
        # Fetch news from GDELT API
        csv_url = (
            f"https://api.gdeltproject.org/api/v2/doc/doc?"
            f"query=(Tesla%20OR%20%22Electric%20vehicles%22%20OR%20%22Automotive%20industry%22%20OR%20EVs%20OR%20Rivian%20OR%20%22Lucid%20Motors%22%20OR%20Ford%20OR%20GM%20OR%20BYD)"
            f"&startdatetime={date}000000"
            f"&enddatetime={date}235959"
            f"&maxrecords=250"
            f"&format=csv"
        )

        try:
            df = pd.read_csv(csv_url)
            df.columns = df.columns.str.lower()
            if 'documentidentifier' in df.columns:
                df['title'] = df['documentidentifier']
            return df
        except Exception as e:
            print("Error fetching news:", e)
            return pd.DataFrame()

    def get_and_classify_news(self, date):
        df = self.get_from_internet(date)
        if df.empty:
            return df
        df['relevance'] = df['title'].astype(str).apply(self.predict)
        return df

# === Test block ===
if __name__ == "__main__":
    predictor = TeslaPredictor()
    
    # Test single sentence
    print(predictor.predict("Tesla unveils new model"))
    print(predictor.predict("Microsoft acquires startup"))

    # Get and classify daily news
    df_labeled = predictor.get_and_classify_news("20250429")
    df_labeled.to_csv('output.csv', index=False)
    print(df_labeled[['title', 'relevance']].head())
