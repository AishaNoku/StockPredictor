import pandas as pd
import glob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import joblib


model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

joblib.dump(model, "finbert_model.joblib")

joblib.dump(tokenizer, "finbert_tokenizer.joblib")


def get_sentiment_probabilities(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
    return probs  # [positive, neutral, negative]

csv_files = glob.glob("./tesla_news/*.csv")
print(csv_files)


count = 1
for file in csv_files:
    df = pd.read_csv(file)
    df["Sentiment_Probs"] = df["Title"].apply(get_sentiment_probabilities)

    
    df[["Positive", "Neutral", "Negative"]] = pd.DataFrame(df["Sentiment_Probs"].tolist(), index=df.index)

    df.drop(columns=["Sentiment_Probs"], inplace=True)

    df.to_csv(file, index=False)

    print(df.head())
    print(f"Processed file number: {count}")
    count += 1
