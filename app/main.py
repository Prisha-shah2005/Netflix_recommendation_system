"""
Netflix Recommendation System - FastAPI Backend
Author: Senior Machine Learning Engineer / Data Scientist

This script implements the FastAPI backend for our recommendation system.
Endpoints:
- `/`: Renders the index.html frontend.
- `/api/users`: Returns top 10 active demo users sorted by watch frequency.
- `/api/user/{user_id}/history`: Returns the watch list, search history, and likes/dislikes for a user.
- `/api/user/{user_id}/recommendations`: Returns hybrid recommendations for a user.
- `/api/movie/{movie_id}`: Returns details of a movie and similar titles.
"""

import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# Add root directory to python path for importing modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.recommendation.hybrid_model import HybridRecommender

app = FastAPI(title="Netflix Recommendation System API", version="1.0.0")

# Paths
base_dir = r"d:\project\Netflix Recommendation System"
cleaned_dir = os.path.join(base_dir, "data", "cleaned")
static_dir = os.path.join(base_dir, "app", "static")
templates_dir = os.path.join(base_dir, "app", "templates")

# Ensure app directories exist
os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

# Mount static files and setup templates
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Global variables to cache loaded dataframes and models
movies_df = None
watch_df = None
search_df = None
reviews_df = None
user_features_df = None
recommender = None

@app.on_event("startup")
def startup_event():
    global movies_df, watch_df, search_df, reviews_df, user_features_df, recommender
    print("Initializing backend, loading datasets, and fitting hybrid model...")
    
    # Load cleaned files
    movies_df = pd.read_csv(os.path.join(cleaned_dir, "movies.csv"))
    watch_df = pd.read_csv(os.path.join(cleaned_dir, "watch_history.csv"))
    search_df = pd.read_csv(os.path.join(cleaned_dir, "search_logs.csv"))
    reviews_df = pd.read_csv(os.path.join(cleaned_dir, "reviews.csv"))
    user_features_df = pd.read_csv(os.path.join(cleaned_dir, "engineered_user_features.csv"))
    
    # Instantiate and fit recommender
    recommender = HybridRecommender(cleaned_dir=cleaned_dir)
    recommender.fit()
    print("Startup complete. System ready!")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """
    Serves the main dashboard page.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/users")
def get_users():
    """
    Returns the top 10 most active demo users based on watch history.
    This provides a varied set of users with rich histories for frontend demonstration.
    """
    global user_features_df
    # Sort users by watch frequency and take the top 10
    top_users = user_features_df.sort_values(by="watch_frequency", ascending=False).head(15)
    
    users_list = []
    for _, row in top_users.iterrows():
        # Find favorite genre (highest preference score column)
        pref_cols = [col for col in user_features_df.columns if col.startswith("genre_pref_")]
        fav_genre_col = row[pref_cols].astype(float).idxmax()
        fav_genre = fav_genre_col.replace("genre_pref_", "").replace("_", " ").title()
        
        users_list.append({
            "user_id": row["user_id"],
            "watch_frequency": int(row["watch_frequency"]),
            "search_frequency": int(row["search_frequency"]),
            "favorite_genre": fav_genre
        })
    return users_list

@app.get("/api/user/{user_id}/history")
def get_user_history(user_id: str):
    """
    Returns the combined watch, search, and review history for a user.
    """
    global watch_df, search_df, reviews_df, movies_df
    
    # 1. Watch History (joined with movies details)
    user_watch = watch_df[watch_df["user_id"] == user_id].copy()
    user_watch = user_watch.merge(movies_df[["movie_id", "title", "genre_primary", "content_type", "language", "duration_minutes"]], on="movie_id", how="left")
    # Sort by date descending
    user_watch = user_watch.sort_values(by="watch_date", ascending=False)
    
    watch_history = []
    for _, row in user_watch.iterrows():
        watch_history.append({
            "movie_id": row["movie_id"],
            "title": row["title"],
            "genre": row["genre_primary"],
            "content_type": row["content_type"],
            "language": row["language"],
            "watch_date": row["watch_date"],
            "progress_percentage": float(row["progress_percentage"]),
            "watch_duration_minutes": float(row["watch_duration_minutes"]),
            "movie_duration": float(row["duration_minutes"]),
            "action": row["action"],
            "rating": int(row["user_rating"]) if pd.notnull(row["user_rating"]) else None
        })
        
    # 2. Search History
    user_search = search_df[search_df["user_id"] == user_id].copy()
    user_search = user_search.sort_values(by="search_date", ascending=False)
    
    search_history = []
    for _, row in user_search.iterrows():
        search_history.append({
            "query": row["search_query"],
            "date": row["search_date"],
            "results_returned": int(row["results_returned"]),
            "clicked_position": int(row["clicked_result_position"])
        })
        
    # 3. Reviews (Likes/Dislikes)
    user_reviews = reviews_df[reviews_df["user_id"] == user_id].copy()
    user_reviews = user_reviews.merge(movies_df[["movie_id", "title", "genre_primary"]], on="movie_id", how="left")
    user_reviews = user_reviews.sort_values(by="review_date", ascending=False)
    
    ratings_history = []
    for _, row in user_reviews.iterrows():
        ratings_history.append({
            "movie_id": row["movie_id"],
            "title": row["title"],
            "genre": row["genre_primary"],
            "rating": int(row["rating"]),
            "sentiment": row["sentiment"],
            "review_text": row["review_text"],
            "date": row["review_date"]
        })
        
    # Calculate quick stats
    fav_genre = "N/A"
    total_hours = 0.0
    if watch_history:
        genres = [item["genre"] for item in watch_history if pd.notnull(item["genre"])]
        if genres:
            fav_genre = max(set(genres), key=genres.count)
        total_hours = sum(item["watch_duration_minutes"] for item in watch_history) / 60.0
        
    return {
        "user_id": user_id,
        "stats": {
            "total_watch_hours": round(total_hours, 1),
            "watch_count": len(watch_history),
            "search_count": len(search_history),
            "rating_count": len(ratings_history),
            "favorite_genre": fav_genre
        },
        "watch_history": watch_history,
        "search_history": search_history,
        "ratings_history": ratings_history
    }

@app.get("/api/user/{user_id}/recommendations")
def get_recommendations(user_id: str):
    """
    Generates personalized recommendations using our dynamic hybrid recommender.
    """
    global recommender
    try:
        recommendations = recommender.recommend(user_id, top_n=10)
        
        recs_list = []
        for rank, (movie_id, score) in enumerate(recommendations, 1):
            details = recommender.get_movie_details(movie_id)
            if details:
                recs_list.append({
                    "rank": rank,
                    "movie_id": movie_id,
                    "title": details["title"],
                    "content_type": details["content_type"],
                    "genre_primary": details["genre_primary"],
                    "genre_secondary": details["genre_secondary"],
                    "release_year": int(details["release_year"]),
                    "duration_minutes": float(details["duration_minutes"]),
                    "rating": details["rating"],
                    "language": details["language"],
                    "country": details["country_of_origin"],
                    "imdb_rating": float(details["imdb_rating"]),
                    "is_netflix_original": bool(details["is_netflix_original"]),
                    "match_score": round(score * 100, 1) # represent as matching percentage
                })
        return recs_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation engine error: {str(e)}")

@app.get("/api/movie/{movie_id}")
def get_movie_details(movie_id: str):
    """
    Returns movie metadata and similar items ("More Like This" list).
    """
    global recommender
    details = recommender.get_movie_details(movie_id)
    if not details:
        raise HTTPException(status_code=404, detail="Movie not found in catalog")
        
    similar_movies = recommender.content_model.get_similar_movies(movie_id, top_n=6)
    similar_list = []
    
    for sm_id, sim_score in similar_movies:
        sm_details = recommender.get_movie_details(sm_id)
        if sm_details:
            similar_list.append({
                "movie_id": sm_id,
                "title": sm_details["title"],
                "content_type": sm_details["content_type"],
                "genre_primary": sm_details["genre_primary"],
                "imdb_rating": float(sm_details["imdb_rating"]),
                "release_year": int(sm_details["release_year"]),
                "language": sm_details["language"],
                "is_netflix_original": bool(sm_details["is_netflix_original"]),
                "similarity_score": round(sim_score * 100, 1)
            })
            
    # Cast formatting details
    movie_info = {
        "movie_id": details["movie_id"],
        "title": details["title"],
        "content_type": details["content_type"],
        "genre_primary": details["genre_primary"],
        "genre_secondary": details["genre_secondary"],
        "release_year": int(details["release_year"]),
        "duration_minutes": float(details["duration_minutes"]),
        "rating": details["rating"],
        "language": details["language"],
        "country": details["country_of_origin"],
        "imdb_rating": float(details["imdb_rating"]),
        "is_netflix_original": bool(details["is_netflix_original"]),
        "added_to_platform": details["added_to_platform"],
        "content_warning": bool(details["content_warning"]),
        "similar_movies": similar_list
    }
    return movie_info
