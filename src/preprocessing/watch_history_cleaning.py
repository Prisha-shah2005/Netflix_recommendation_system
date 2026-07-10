"""
Netflix Recommendation System - Watch History Dataset Cleaning Script
Author: Senior Machine Learning Engineer / Data Scientist

This script cleans the raw `watch_history.csv` dataset.
Cleaning Steps:
1. Load dataset from `data/raw/watch_history.csv`.
2. Inspect shape, columns, and duplicates.
3. Remove exact duplicate sessions based on `session_id`.
4. Standardize text columns (device_type, action, quality, location_country).
5. Standardize `watch_date` to YYYY-MM-DD and validate against the current date.
6. Join with `data/cleaned/movies.csv` to fetch total movie duration for filling missing metrics.
7. Fill missing `watch_duration_minutes` and `progress_percentage` using their mathematical relationship.
8. Resolve logical anomalies: cap progress_percentage at 100.0% and watch_duration_minutes at the movie's total duration.
9. Keep track of sparse `user_rating` (fill missing with NaN).
10. Save the cleaned dataset to `data/cleaned/watch_history.csv`.
"""

import os
import pandas as pd
import numpy as np

def clean_watch_history(raw_path, cleaned_path, movies_clean_path, current_date="2026-07-10"):
    print("Starting Watch History dataset cleaning...")
    
    # 1. Load datasets
    df = pd.read_csv(raw_path)
    initial_shape = df.shape
    print(f"Initial shape: {initial_shape}")
    
    # Load cleaned movies to merge duration details
    if not os.path.exists(movies_clean_path):
        raise FileNotFoundError(f"Cleaned movies file not found at {movies_clean_path}. Clean movies first.")
    movies_df = pd.read_csv(movies_clean_path)[["movie_id", "duration_minutes"]]
    movies_df = movies_df.rename(columns={"duration_minutes": "movie_total_duration"})
    
    # 2. Duplicate removal
    # session_id must be unique
    df = df.drop_duplicates(subset=["session_id"], keep="first")
    print(f"Shape after removing duplicates: {df.shape} (Removed {initial_shape[0] - df.shape[0]} sessions)")
    
    # 3. Standardize text columns
    text_cols = ["device_type", "action", "quality", "location_country"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan})
            
    # 4. Standardize and validate dates
    df["watch_date"] = pd.to_datetime(df["watch_date"], errors="coerce")
    if df["watch_date"].isnull().any():
        date_mode = df["watch_date"].mode()[0]
        df["watch_date"] = df["watch_date"].fillna(date_mode)
        
    current_dt = pd.to_datetime(current_date)
    future_dates_mask = df["watch_date"] > current_dt
    if future_dates_mask.any():
        print(f"Capping {future_dates_mask.sum()} future 'watch_date' records to current date {current_date}.")
        df.loc[future_dates_mask, "watch_date"] = current_dt
        
    df["watch_date"] = df["watch_date"].dt.strftime("%Y-%m-%d")
    
    # 5. Merge movie duration metadata to resolve missing values mathematically
    df = df.merge(movies_df, on="movie_id", how="left")
    
    # For any movie_id not matching, impute movie_total_duration with global median
    global_movie_dur = df["movie_total_duration"].median()
    df["movie_total_duration"] = df["movie_total_duration"].fillna(global_movie_dur)
    
    # 6. Fill missing watch_duration_minutes and progress_percentage mathematically
    # Formula: watch_duration = (progress_percentage / 100) * movie_total_duration
    # Formula: progress_percentage = (watch_duration / movie_total_duration) * 100
    
    # Case A: watch_duration is null, progress_percentage is not null
    mask_a = df["watch_duration_minutes"].isnull() & df["progress_percentage"].notnull()
    df.loc[mask_a, "watch_duration_minutes"] = (df.loc[mask_a, "progress_percentage"] / 100.0) * df.loc[mask_a, "movie_total_duration"]
    
    # Case B: progress_percentage is null, watch_duration is not null
    mask_b = df["progress_percentage"].isnull() & df["watch_duration_minutes"].notnull()
    df.loc[mask_b, "progress_percentage"] = (df.loc[mask_b, "watch_duration_minutes"] / df.loc[mask_b, "movie_total_duration"]) * 100.0
    
    # Case C: Both are null. Impute progress based on 'action' status, then calculate duration
    mask_c = df["watch_duration_minutes"].isnull() & df["progress_percentage"].isnull()
    print(f"Resolving {mask_c.sum()} records where both watch duration and progress percentage are missing.")
    
    # Map actions to default progress percentages:
    # 'completed' -> 100%, 'paused' -> 50%, 'stopped' -> 45%, 'started' -> 10%
    action_progress_map = {
        "completed": 100.0,
        "paused": 50.0,
        "stopped": 45.0,
        "started": 10.0
    }
    for action, default_progress in action_progress_map.items():
        action_mask = mask_c & (df["action"] == action)
        df.loc[action_mask, "progress_percentage"] = default_progress
        df.loc[action_mask, "watch_duration_minutes"] = (default_progress / 100.0) * df.loc[action_mask, "movie_total_duration"]
        
    # Any remaining nulls after mapping
    df["progress_percentage"] = df["progress_percentage"].fillna(50.0)
    df["watch_duration_minutes"] = df["watch_duration_minutes"].fillna(df["movie_total_duration"] * 0.5)
    
    # 7. Logical Capping / Outlier Resolution
    # Progress percentage cannot exceed 100.0%
    over_100_progress = df["progress_percentage"] > 100.0
    if over_100_progress.any():
        df.loc[over_100_progress, "progress_percentage"] = 100.0
        
    # Watch duration cannot exceed the movie's total duration
    over_duration = df["watch_duration_minutes"] > df["movie_total_duration"]
    if over_duration.any():
        df.loc[over_duration, "watch_duration_minutes"] = df.loc[over_duration, "movie_total_duration"]
        
    # Standardize precision to 1 decimal place
    df["progress_percentage"] = df["progress_percentage"].round(1)
    df["watch_duration_minutes"] = df["watch_duration_minutes"].round(1)
    
    # 8. User Ratings
    # user_rating is highly sparse (79.91% missing). We preserve it as NaN representing unrated items.
    # No imputation should be performed to avoid introducing false positive/negative bias.
    # Just round it if there are floats.
    df["user_rating"] = df["user_rating"].round(0)
    
    # 9. Clean up temporary join columns
    df = df.drop(columns=["movie_total_duration"])
    
    # 10. Final validation and save
    print(f"Cleaned shape: {df.shape}")
    print(f"Null count in cleaned watch history (excluding user_rating): {df.drop(columns=['user_rating']).isnull().sum().sum()}")
    
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned watch history to {cleaned_path}\n")
    return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_csv = os.path.join(base_dir, "data", "raw", "watch_history.csv")
    cleaned_csv = os.path.join(base_dir, "data", "cleaned", "watch_history.csv")
    movies_clean_csv = os.path.join(base_dir, "data", "cleaned", "movies.csv")
    clean_watch_history(raw_csv, cleaned_csv, movies_clean_csv)
