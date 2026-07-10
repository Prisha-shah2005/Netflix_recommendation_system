"""
Netflix Recommendation System - Reviews Dataset Cleaning Script
Author: Senior Machine Learning Engineer / Data Scientist

This script cleans the raw `reviews.csv` dataset.
Cleaning Steps:
1. Load dataset from `data/raw/reviews.csv`.
2. Inspect shape, columns, and duplicates.
3. Remove duplicate rows based on `review_id`.
4. Standardize text columns (device_type, sentiment).
5. Standardize `review_date` to YYYY-MM-DD and validate against the current date.
6. Impute missing `helpful_votes` and `total_votes` with 0.
7. Fill missing `review_text` with empty strings.
8. Impute missing `sentiment_score` dynamically by calculating the average score
   for each sentiment category ('positive', 'neutral', 'negative') from non-null rows.
9. Save the cleaned dataset to `data/cleaned/reviews.csv`.
"""

import os
import pandas as pd
import numpy as np

def clean_reviews(raw_path, cleaned_path, current_date="2026-07-10"):
    print("Starting Reviews dataset cleaning...")
    
    # 1. Load dataset
    df = pd.read_csv(raw_path)
    initial_shape = df.shape
    print(f"Initial shape: {initial_shape}")
    
    # 2. Duplicate removal
    df = df.drop_duplicates(subset=["review_id"], keep="first")
    print(f"Shape after removing duplicates: {df.shape} (Removed {initial_shape[0] - df.shape[0]} reviews)")
    
    # 3. Standardize text columns
    text_cols = ["device_type", "sentiment"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({"nan": np.nan, "none": np.nan})
            
    # Normalize review_text
    df["review_text"] = df["review_text"].fillna("No Review Text").astype(str).str.strip()
    # Replace actual blank values
    df.loc[df["review_text"] == "", "review_text"] = "No Review Text"
    
    # 4. Standardize and validate dates
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    if df["review_date"].isnull().any():
        date_mode = df["review_date"].mode()[0]
        df["review_date"] = df["review_date"].fillna(date_mode)
        
    current_dt = pd.to_datetime(current_date)
    future_dates_mask = df["review_date"] > current_dt
    if future_dates_mask.any():
        print(f"Capping {future_dates_mask.sum()} future 'review_date' records to current date {current_date}.")
        df.loc[future_dates_mask, "review_date"] = current_dt
        
    df["review_date"] = df["review_date"].dt.strftime("%Y-%m-%d")
    
    # 5. Impute vote statistics
    # When helpful_votes or total_votes is missing, we assume 0 votes were cast (default)
    df["helpful_votes"] = df["helpful_votes"].fillna(0.0).astype(int)
    df["total_votes"] = df["total_votes"].fillna(0.0).astype(int)
    
    # Ensure logical integrity: helpful_votes cannot exceed total_votes
    invalid_votes_mask = df["helpful_votes"] > df["total_votes"]
    if invalid_votes_mask.any():
        print(f"Fixing {invalid_votes_mask.sum()} records where helpful_votes exceeded total_votes.")
        df.loc[invalid_votes_mask, "total_votes"] = df.loc[invalid_votes_mask, "helpful_votes"]
        
    # 6. Impute sentiment_score dynamically
    # Calculate the average sentiment_score for positive, neutral, and negative classes from non-null rows
    sentiment_means = df.groupby("sentiment")["sentiment_score"].mean().to_dict()
    
    # Fallback to defaults in case a category is completely null (rare)
    default_means = {"positive": 0.82, "neutral": 0.51, "negative": 0.18}
    for key, default_val in default_means.items():
        if key not in sentiment_means or pd.isnull(sentiment_means[key]):
            sentiment_means[key] = default_val
            
    print(f"Dynamic sentiment score imputation means: {sentiment_means}")
    
    # Impute missing sentiment_scores using mapped category means
    null_sentiment_mask = df["sentiment_score"].isnull()
    if null_sentiment_mask.any():
        print(f"Imputing {null_sentiment_mask.sum()} missing sentiment_score values using category means.")
        mapped_means = df.loc[null_sentiment_mask, "sentiment"].map(sentiment_means)
        df.loc[null_sentiment_mask, "sentiment_score"] = mapped_means
        
    df["sentiment_score"] = df["sentiment_score"].round(3)
    
    # 7. Final verification and save
    print(f"Cleaned shape: {df.shape}")
    print(f"Null count in cleaned reviews: {df.isnull().sum().sum()}")
    
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned reviews to {cleaned_path}\n")
    return df

if __name__ == "__main__":
    raw_csv = r"d:\project\Netflix Recommendation System\data\raw\reviews.csv"
    cleaned_csv = r"d:\project\Netflix Recommendation System\data\cleaned\reviews.csv"
    clean_reviews(raw_csv, cleaned_csv)
