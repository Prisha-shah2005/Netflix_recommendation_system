"""
Netflix Recommendation System - Content-Based Recommendation Engine
Author: Senior Machine Learning Engineer / Data Scientist

This script implements the ContentBasedRecommender class which:
1. Loads engineered movie metadata features (TF-IDF) and interaction histories.
2. Constructs a User Profile Vector by taking the weighted average of TF-IDF vectors of movies the user has watched.
3. Computes the Cosine Similarity between the User Profile Vector and all candidate movies.
4. Generates personalized content-based recommendations by filtering out already watched movies.
5. Computes Item-to-Item Similarity to support "More Like This" query options on the frontend.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.data_loader import load_cached_csv

class ContentBasedRecommender:
    def __init__(self, cleaned_dir=None):
        if cleaned_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cleaned_dir = os.path.join(base_dir, "data", "cleaned")
        self.cleaned_dir = cleaned_dir
        self.movies_features_path = os.path.join(cleaned_dir, "engineered_movie_features.csv")
        self.interactions_path = os.path.join(cleaned_dir, "engineered_interactions.csv")
        
        self.movie_features_df = None
        self.interactions_df = None
        
        # Extracted matrices for computations
        self.movie_ids = None
        self.tfidf_matrix = None
        self.movie_id_to_idx = None
        
    def fit(self):
        """
        Load features and pre-compute similarities.
        """
        print("Fitting Content-Based Recommender...")
        self.movie_features_df = load_cached_csv(self.movies_features_path)
        self.interactions_df = load_cached_csv(self.interactions_path)
        
        self.movie_ids = self.movie_features_df["movie_id"].values
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        
        # Extract TF-IDF feature columns (columns starting with 'tfidf_')
        tfidf_cols = [col for col in self.movie_features_df.columns if col.startswith("tfidf_")]
        self.tfidf_matrix = self.movie_features_df[tfidf_cols].values
        print(f"Content-Based Recommender fitted. Catalog size: {self.tfidf_matrix.shape[0]} movies, TF-IDF features: {self.tfidf_matrix.shape[1]}")
        
    def _build_user_profile(self, user_id):
        """
        Builds a user profile vector by aggregating TF-IDF vectors of movies the user watched,
        weighted by their interaction score.
        """
        user_history = self.interactions_df[self.interactions_df["user_id"] == user_id]
        if user_history.empty:
            # Cold start user: return a zero vector
            return np.zeros(self.tfidf_matrix.shape[1])
            
        # Get movie indices and weights
        movie_indices = []
        weights = []
        
        for _, row in user_history.iterrows():
            mid = row["movie_id"]
            weight = row["interaction_score"]
            if mid in self.movie_id_to_idx:
                movie_indices.append(self.movie_id_to_idx[mid])
                weights.append(weight)
                
        if not movie_indices:
            return np.zeros(self.tfidf_matrix.shape[1])
            
        # Extract vectors and calculate weighted average
        user_vectors = self.tfidf_matrix[movie_indices]
        weights = np.array(weights).reshape(-1, 1)
        
        # Weighted profile vector
        weighted_profile = np.sum(user_vectors * weights, axis=0)
        norm = np.linalg.norm(weighted_profile)
        if norm > 0:
            weighted_profile = weighted_profile / norm
            
        return weighted_profile

    def recommend(self, user_id, top_n=10, exclude_watched=True):
        """
        Recommend top_n movies for user_id.
        """
        user_profile = self._build_user_profile(user_id)
        
        # If cold-start user (profile vector is all zeros), return popular movies as fallback
        if np.all(user_profile == 0.0):
            # Sort by imdb rating and popularity
            popular = self.movie_features_df.sort_values(by=["imdb_rating"], ascending=False).head(top_n)
            return list(zip(popular["movie_id"].values, [0.5] * top_n))
            
        # Compute cosine similarity between user profile and all movies
        similarities = cosine_similarity(user_profile.reshape(1, -1), self.tfidf_matrix).flatten()
        
        # Create a mapping of movie_id to similarity score
        scores = list(zip(self.movie_ids, similarities))
        
        if exclude_watched:
            # Get movies already watched by the user
            watched_movies = set(self.interactions_df[self.interactions_df["user_id"] == user_id]["movie_id"].values)
            scores = [(mid, score) for mid, score in scores if mid not in watched_movies]
            
        # Sort by similarity score descending
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores[:top_n]
        
    def get_similar_movies(self, movie_id, top_n=10):
        """
        Supports 'More Like This' feature. Calculates item-item similarity.
        """
        if movie_id not in self.movie_id_to_idx:
            print(f"Warning: Movie ID {movie_id} not found in catalog.")
            return []
            
        idx = self.movie_id_to_idx[movie_id]
        movie_vector = self.tfidf_matrix[idx].reshape(1, -1)
        
        # Compute similarity between this movie and all catalog movies
        similarities = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()
        
        scores = list(zip(self.movie_ids, similarities))
        # Exclude the movie itself
        scores = [(mid, score) for mid, score in scores if mid != movie_id]
        
        # Sort and return top_n
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores[:top_n]

if __name__ == "__main__":
    recommender = ContentBasedRecommender()
    recommender.fit()
    # Test recommendations
    test_user = "user_07271"
    recs = recommender.recommend(test_user, top_n=5)
    print(f"\nPersonalized recommendations for {test_user}:")
    for mid, score in recs:
        title = recommender.movie_features_df[recommender.movie_features_df["movie_id"] == mid]["title"].values[0]
        print(f"  - {title} (ID: {mid}, score: {score:.3f})")
        
    # Test item similarity
    test_movie = "movie_0001"
    similar = recommender.get_similar_movies(test_movie, top_n=3)
    movie_title = recommender.movie_features_df[recommender.movie_features_df["movie_id"] == test_movie]["title"].values[0]
    print(f"\nMore like '{movie_title}':")
    for mid, score in similar:
        title = recommender.movie_features_df[recommender.movie_features_df["movie_id"] == mid]["title"].values[0]
        print(f"  - {title} (ID: {mid}, score: {score:.3f})")
