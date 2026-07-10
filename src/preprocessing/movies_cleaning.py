"""
Netflix Recommendation System - Movies Dataset Cleaning Script
Author: Senior Machine Learning Engineer / Data Scientist

This script cleans the raw `movies.csv` dataset.
Cleaning Steps:
1. Load dataset from `data/raw/movies.csv`.
2. Inspect shape, columns, and duplicates.
3. Remove exact duplicate rows and duplicate `movie_id` keys to ensure primary key uniqueness.
4. Drop redundant and heavily missing columns ('production_budget', 'box_office_revenue')
   with detailed justification (over 65% missing and irrelevant for behavior-based personalization).
5. Standardize text columns (title, content_type, genres, rating, language, country) to strip whitespaces.
6. Impute missing `genre_secondary` with a placeholder ('None').
7. Impute missing `imdb_rating` with the median rating of its content type and primary genre,
   falling back to content type median, then global median if necessary.
8. Resolve duration anomalies: replace duration_minutes <= 0.0 with the median duration of its content type.
9. Impute TV-series specific fields (number_of_seasons, number_of_episodes) with 0 for movies/stand-up.
10. Standardize `added_to_platform` to YYYY-MM-DD format and validate that no dates are in the future.
11. Save the cleaned dataset to `data/cleaned/movies.csv`.
"""

import os
import pandas as pd
import numpy as np

def clean_movies(raw_path, cleaned_path, current_date="2026-07-10"):
    print("Starting Movies dataset cleaning...")
    
    # 1. Load raw dataset
    df = pd.read_csv(raw_path)
    initial_shape = df.shape
    print(f"Initial shape: {initial_shape}")
    
    # 2. Duplicate analysis & removal
    # We must ensure movie_id is unique to act as a proper primary key in our database
    df = df.drop_duplicates(subset=["movie_id"], keep="first")
    print(f"Shape after removing duplicates: {df.shape} (Removed {initial_shape[0] - df.shape[0]} rows)")
    
    # 3. Column Importance Analysis & Drop Unnecessary Columns
    # 'production_budget' and 'box_office_revenue' have >64% missing values. More importantly,
    # financial metrics are not helpful for behavioral recommendation algorithms (collaborative
    # or content-based) since user preference is driven by genre, content type, language, and quality,
    # not the movie's budget.
    cols_to_drop = ["production_budget", "box_office_revenue"]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    print(f"Dropped columns: {cols_to_drop} due to high sparsity and low features significance.")
    
    # 4. Standardize text columns
    text_cols = ["title", "content_type", "genre_primary", "genre_secondary", 
                 "rating", "language", "country_of_origin"]
    for col in text_cols:
        if col in df.columns:
            # Strip trailing/leading whitespaces
            df[col] = df[col].astype(str).str.strip()
            # Replace string representations of nan/None
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan})
            
    # 5. Handle missing values for categorical features
    df["genre_secondary"] = df["genre_secondary"].fillna("No Secondary Genre")
    
    # 6. Impute missing imdb_rating using grouped medians
    # A movie's IMDb rating can be approximated by the median rating of its content type and primary genre.
    # Group by content_type and genre_primary to compute medians
    grouped_medians = df.groupby(["content_type", "genre_primary"])["imdb_rating"].transform("median")
    df["imdb_rating"] = df["imdb_rating"].fillna(grouped_medians)
    
    # Fallback 1: group by content_type only
    type_medians = df.groupby("content_type")["imdb_rating"].transform("median")
    df["imdb_rating"] = df["imdb_rating"].fillna(type_medians)
    
    # Fallback 2: global median
    global_median = df["imdb_rating"].median()
    df["imdb_rating"] = df["imdb_rating"].fillna(global_median)
    
    # 7. Outlier and anomaly inspection: duration_minutes
    # Movies cannot have a duration of 0. We impute duration_minutes <= 0 with the median duration of its content type.
    invalid_duration_mask = df["duration_minutes"] <= 0
    if invalid_duration_mask.any():
        print(f"Found {invalid_duration_mask.sum()} rows with invalid duration (<= 0). Imputing with content-type medians.")
        content_type_durations = df.groupby("content_type")["duration_minutes"].transform("median")
        df.loc[invalid_duration_mask, "duration_minutes"] = content_type_durations[invalid_duration_mask]
    
    # Round duration to 1 decimal place
    df["duration_minutes"] = df["duration_minutes"].round(1)

    # 8. TV-Series specific imputation
    # Movies and stand-ups don't have seasons or episodes. Let's fill them with 0.
    # For TV Series / Limited Series, if they are missing, we fill with median values of series.
    is_series = df["content_type"].isin(["TV Series", "Limited Series"])
    
    df.loc[~is_series, "number_of_seasons"] = 0.0
    df.loc[~is_series, "number_of_episodes"] = 0.0
    
    # Impute missing series values with their content type group medians
    df.loc[is_series, "number_of_seasons"] = df.loc[is_series, "number_of_seasons"].fillna(
        df[df["content_type"] == "TV Series"]["number_of_seasons"].median()
    )
    df.loc[is_series, "number_of_episodes"] = df.loc[is_series, "number_of_episodes"].fillna(
        df[df["content_type"] == "TV Series"]["number_of_episodes"].median()
    )

    # 9. Standardize and validate dates
    # Parse added_to_platform as datetime
    df["added_to_platform"] = pd.to_datetime(df["added_to_platform"], errors="coerce")
    
    # Fill missing dates with the most common date (mode) or forward fill
    if df["added_to_platform"].isnull().any():
        date_mode = df["added_to_platform"].mode()[0]
        df["added_to_platform"] = df["added_to_platform"].fillna(date_mode)
        
    # Validate date range: ensure no release/addition dates are in the future
    current_dt = pd.to_datetime(current_date)
    future_dates_mask = df["added_to_platform"] > current_dt
    if future_dates_mask.any():
        print(f"Capping {future_dates_mask.sum()} future 'added_to_platform' dates to current date {current_date}.")
        df.loc[future_dates_mask, "added_to_platform"] = current_dt
        
    # Convert dates back to standard string format YYYY-MM-DD
    df["added_to_platform"] = df["added_to_platform"].dt.strftime("%Y-%m-%d")
    
    # 10. Final verification and save
    print(f"Cleaned shape: {df.shape}")
    print(f"Null count in cleaned movies: {df.isnull().sum().sum()}")
    
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned movies to {cleaned_path}\n")
    return df

if __name__ == "__main__":
    raw_csv = r"d:\project\Netflix Recommendation System\data\raw\movies.csv"
    cleaned_csv = r"d:\project\Netflix Recommendation System\data\cleaned\movies.csv"
    clean_movies(raw_csv, cleaned_csv)
