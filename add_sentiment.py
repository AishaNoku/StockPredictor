import pandas as pd
import glob
import numpy as np
import os
import re
from datetime import datetime

def normalize_values(row):
    """Normalize values to ensure they sum to 1"""
    total = row['Positive'] + row['Neutral'] + row['Negative']
    if total > 0:
        return pd.Series({
            'Positive': row['Positive'] / total,
            'Neutral': row['Neutral'] / total,
            'Negative': row['Negative'] / total
        })
    return pd.Series({'Positive': 0, 'Neutral': 0, 'Negative': 0})

def extract_date_from_filename(filename):
    """Extract date from filename"""
    # Get just the filename without path
    base_filename = os.path.basename(filename)
    
    # Try to extract date using regex (looking for patterns like YYYY-MM-DD or similar)
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', base_filename)
    if date_match:
        return date_match.group(1)
    
    # If the pattern above doesn't match, try other common formats
    # For format like "tesla_news_20240419.csv"
    date_match = re.search(r'(\d{8})', base_filename)
    if date_match:
        date_str = date_match.group(1)
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    # If no match found, return None
    return None

def main():
    # Path to the CSV files with sentiment analysis
    csv_files = glob.glob("./tesla_news/*.csv")
    print(f"Found {len(csv_files)} files to process")
    
    # Create an empty list to store all data
    all_data = []
    
    # Process each file
    for i, file in enumerate(csv_files, 1):
        try:
            # Extract date from filename
            file_date = extract_date_from_filename(file)
            if not file_date:
                print(f"Warning: Could not extract date from filename {file}, skipping...")
                continue
                
            print(f"Processing file {i}/{len(csv_files)}: {file} (Date: {file_date})")
            
            # Read the CSV file
            df = pd.read_csv(file)
            
            # Check if sentiment columns exist
            if not all(col in df.columns for col in ['Positive', 'Neutral', 'Negative']):
                print(f"Warning: Missing sentiment columns in {file}, skipping...")
                continue
            
            # Add the date from filename to all rows
            df['Date'] = file_date
            
            # Select only necessary columns
            df = df[['Date', 'Positive', 'Neutral', 'Negative']]
            
            # Append to all_data
            all_data.append(df)
            
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # Combine all data
    if not all_data:
        print("No valid data found in the files.")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")
    
    # Group by date and calculate average sentiment
    daily_sentiment = combined_df.groupby('Date').agg({
        'Positive': 'mean',
        'Neutral': 'mean',
        'Negative': 'mean'
    }).reset_index()
    
    # Normalize sentiment values to ensure they sum to 1
    sentiment_cols = ['Positive', 'Neutral', 'Negative']
    daily_sentiment[sentiment_cols] = daily_sentiment.apply(normalize_values, axis=1)
    
    # Calculate overall sentiment score (-1 to 1 scale)
    daily_sentiment['Sentiment_Score'] = daily_sentiment['Positive'] - daily_sentiment['Negative']
    
    # Sort by date
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'], errors="coerce")
    daily_sentiment = daily_sentiment.sort_values('Date')
    daily_sentiment['Date'] = daily_sentiment['Date'].dt.strftime('%Y-%m-%d')
    
    print("\nFirst few rows of aggregated daily sentiment:")
    print(daily_sentiment.head())
    
    # Save to CSV
    output_file = 'tesla_daily_sentiment.csv'
    daily_sentiment.to_csv(output_file, index=False)
    print(f"Saved aggregated daily sentiment to {output_file}")
    
    # Generate summary statistics
    print("\nSummary statistics:")
    print(daily_sentiment[['Positive', 'Neutral', 'Negative', 'Sentiment_Score']].describe())

if __name__ == "__main__":
    main()