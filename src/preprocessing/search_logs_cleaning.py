"""
Netflix Recommendation System - Search Logs Dataset Cleaning Script
Author: Senior Machine Learning Engineer / Data Scientist

This script cleans the raw `search_logs.csv` dataset.
Cleaning Steps:
1. Load dataset from `data/raw/search_logs.csv`.
2. Inspect shape, columns, and duplicates.
3. Remove duplicate rows based on `search_id` to ensure unique primary key records.
4. Standardize text columns (device_type, location_country).
5. Standardize `search_query` (convert to lowercase, strip leading/trailing spaces).
6. Impute missing `clicked_result_position` with -1 (representing "no result clicked").
7. Impute missing `search_duration_seconds` with the median search duration.
8. Standardize `search_date` to YYYY-MM-DD and validate against the current date.
9. Save the cleaned dataset to `data/cleaned/search_logs.csv`.
"""

import os
import pandas as pd
import numpy as np

def clean_search_logs(raw_path, cleaned_path, current_date="2026-07-10"):
    print("Starting Search Logs dataset cleaning...")
    
    # 1. Load dataset
    df = pd.read_csv(raw_path)
    initial_shape = df.shape
    print(f"Initial shape: {initial_shape}")
    
    # 2. Duplicate removal
    df = df.drop_duplicates(subset=["search_id"], keep="first")
    print(f"Shape after removing duplicates: {df.shape} (Removed {initial_shape[0] - df.shape[0]} searches)")
    
    # 3. Standardize text columns
    text_cols = ["device_type", "location_country"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan})
            
    # Normalize search queries
    df["search_query"] = df["search_query"].astype(str).str.strip().str.lower()
    
    # 4. Standardize and validate dates
    df["search_date"] = pd.to_datetime(df["search_date"], errors="coerce")
    if df["search_date"].isnull().any():
        date_mode = df["search_date"].mode()[0]
        df["search_date"] = df["search_date"].fillna(date_mode)
        
    current_dt = pd.to_datetime(current_date)
    future_dates_mask = df["search_date"] > current_dt
    if future_dates_mask.any():
        print(f"Capping {future_dates_mask.sum()} future 'search_date' records to current date {current_date}.")
        df.loc[future_dates_mask, "search_date"] = current_dt
        
    df["search_date"] = df["search_date"].dt.strftime("%Y-%m-%d")
    
    # 5. Handle missing values
    # clicked_result_position represents the rank position of the clicked video.
    # When it is missing, it means the search was made but no item was clicked (null).
    # We impute it with -1 to serve as a categorical/numerical indicator for "No Click".
    df["clicked_result_position"] = df["clicked_result_position"].fillna(-1.0)
    df["clicked_result_position"] = df["clicked_result_position"].astype(int)
    
    # Impute missing search duration with median duration (to handle network latency logs that failed to register duration)
    median_duration = df["search_duration_seconds"].median()
    df["search_duration_seconds"] = df["search_duration_seconds"].fillna(median_duration)
    df["search_duration_seconds"] = df["search_duration_seconds"].round(1)
    
    # 6. Final verification and save
    print(f"Cleaned shape: {df.shape}")
    print(f"Null count in cleaned search logs: {df.isnull().sum().sum()}")
    
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned search logs to {cleaned_path}\n")
    return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_csv = os.path.join(base_dir, "data", "raw", "search_logs.csv")
    cleaned_csv = os.path.join(base_dir, "data", "cleaned", "search_logs.csv")
    clean_search_logs(raw_csv, cleaned_csv)
