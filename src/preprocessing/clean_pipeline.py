"""
Netflix Recommendation System - Data Preprocessing Pipeline Coordinator
Author: Senior Machine Learning Engineer / Data Scientist

This script orchestrates the cleaning of all raw datasets:
1. Movies
2. Watch History
3. Search Logs
4. Reviews

It validates the outputs, ensuring no duplicates or unexpected missing values exist,
and outputs a summary of the cleaning process.
"""

import os
import sys
import pandas as pd

# Add src to python path to allow importing preprocessing modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.preprocessing.movies_cleaning import clean_movies
from src.preprocessing.watch_history_cleaning import clean_watch_history
from src.preprocessing.search_logs_cleaning import clean_search_logs
from src.preprocessing.reviews_cleaning import clean_reviews

def run_pipeline():
    print("==================================================")
    
    # Define file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(base_dir, "data", "raw")
    cleaned_dir = os.path.join(base_dir, "data", "cleaned")
    
    movies_raw = os.path.join(raw_dir, "movies.csv")
    movies_clean = os.path.join(cleaned_dir, "movies.csv")
    
    watch_raw = os.path.join(raw_dir, "watch_history.csv")
    watch_clean = os.path.join(cleaned_dir, "watch_history.csv")
    
    search_raw = os.path.join(raw_dir, "search_logs.csv")
    search_clean = os.path.join(cleaned_dir, "search_logs.csv")
    
    reviews_raw = os.path.join(raw_dir, "reviews.csv")
    reviews_clean = os.path.join(cleaned_dir, "reviews.csv")
    
    current_date = "2026-07-10"
    
    print(f"RUNNING DATA CLEANING PIPELINE - TARGET: {cleaned_dir}")
    print("==================================================\n")
    
    # Step 1: Clean Movies
    clean_movies(movies_raw, movies_clean, current_date)
    
    # Step 2: Clean Watch History (Requires Cleaned Movies)
    clean_watch_history(watch_raw, watch_clean, movies_clean, current_date)
    
    # Step 3: Clean Search Logs
    clean_search_logs(search_raw, search_clean, current_date)
    
    # Step 4: Clean Reviews
    clean_reviews(reviews_raw, reviews_clean, current_date)
    
    print("==================================================")
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("==================================================")
    
    # Final Validation Report
    print("\nVALIDATION REPORT:")
    datasets = {
        "Movies": movies_clean,
        "Watch History": watch_clean,
        "Search Logs": search_clean,
        "Reviews": reviews_clean
    }
    
    for name, path in datasets.items():
        df = pd.read_csv(path)
        print(f"\n{name} Dataset:")
        print(f"  - Cleaned Path: {path}")
        print(f"  - Shape: {df.shape}")
        print(f"  - Duplicates: {df.duplicated().sum()}")
        # Check null values (ignoring user_rating in watch_history since it's naturally sparse)
        if name == "Watch History":
            nulls = df.drop(columns=["user_rating"]).isnull().sum().sum()
            print(f"  - Nulls (excluding user_rating): {nulls}")
        else:
            nulls = df.isnull().sum().sum()
            print(f"  - Nulls: {nulls}")

if __name__ == "__main__":
    run_pipeline()
