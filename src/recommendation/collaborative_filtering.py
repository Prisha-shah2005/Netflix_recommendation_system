"""
Netflix Recommendation System - Collaborative Filtering Engine (SVD)
Author: Senior Machine Learning Engineer / Data Scientist

This script implements the CollaborativeFilteringRecommender class which:
1. Loads the User-Movie Interaction Matrix.
2. Normalizes the matrix by subtracting the mean rating of each user (de-meaning).
3. Performs Singular Value Decomposition (SVD) via SciPy's `svds` to extract latent factors.
4. Reconstructs the fully predicted Interaction Matrix.
5. Predicts scores for unobserved user-movie pairs and ranks them for recommendations.
"""

import os
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from src.utils.data_loader import load_cached_csv

class CollaborativeFilteringRecommender:
    def __init__(self, cleaned_dir=None, num_factors=20):
        if cleaned_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cleaned_dir = os.path.join(base_dir, "data", "cleaned")
        self.cleaned_dir = cleaned_dir
        self.matrix_path = os.path.join(cleaned_dir, "interaction_matrix.csv")
        self.interactions_path = os.path.join(cleaned_dir, "engineered_interactions.csv")
        self.movies_path = os.path.join(cleaned_dir, "movies.csv")
        self.num_factors = num_factors
        
        self.interaction_matrix_df = None
        self.interactions_df = None
        self.movies_df = None
        
        # SVD outputs
        self.user_ids = None
        self.movie_ids = None
        self.predicted_ratings_df = None
        
    def fit(self):
        """
        Load interaction matrix, apply SVD, and compute predicted ratings.
        """
        print("Fitting Collaborative Filtering Recommender via SVD...")
        self.interactions_df = load_cached_csv(self.interactions_path)
        self.movies_df = load_cached_csv(self.movies_path)
        
        # Pivot the smaller interactions dataframe to construct the interaction matrix in-memory,
        # which saves memory and avoids MemoryErrors on resource-constrained startup.
        self.interaction_matrix_df = self.interactions_df.pivot(
            index="user_id",
            columns="movie_id",
            values="interaction_score"
        ).fillna(0.0)
        
        self.user_ids = self.interaction_matrix_df.index.values
        self.movie_ids = self.interaction_matrix_df.columns.values
        
        # Convert to numpy matrix
        R = self.interaction_matrix_df.values
        
        # Compute user ratings mean vectorially
        non_zero_counts = np.sum(R > 0.0, axis=1)
        ratings_sum = np.sum(R, axis=1)
        user_ratings_mean = np.zeros(R.shape[0])
        valid_users = non_zero_counts > 0
        user_ratings_mean[valid_users] = ratings_sum[valid_users] / non_zero_counts[valid_users]
                
        # Vectorized de-meaning of the rating matrix
        mask = R > 0.0
        R_demeaned = np.zeros(R.shape)
        R_demeaned[mask] = R[mask] - user_ratings_mean[np.where(mask)[0]]
                    
        # Perform SVD (Singular Value Decomposition)
        k = min(self.num_factors, min(R_demeaned.shape) - 1)
        U, sigma, Vt = svds(R_demeaned, k=k)
        
        # Convert sigma to a diagonal matrix
        sigma = np.diag(sigma)
        
        # Reconstruct the ratings matrix
        all_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        
        # Vectorized adding user rating means back
        all_predicted_ratings = all_predicted_ratings + user_ratings_mean.reshape(-1, 1)
            
        # Clip predicted ratings between [0, 1.2] to match interaction score range
        all_predicted_ratings = np.clip(all_predicted_ratings, 0.0, 1.2)
        
        # Convert to DataFrame
        self.predicted_ratings_df = pd.DataFrame(
            all_predicted_ratings, 
            index=self.user_ids, 
            columns=self.movie_ids
        )
        # Release the raw interaction matrix DataFrame from memory since it's no longer needed
        self.interaction_matrix_df = None
        print(f"Collaborative Filtering Recommender SVD finished. Latent factors: {k}")
        
    def predict_score(self, user_id, movie_id):
        """
        Predict rating score for a single user-movie pair.
        """
        if user_id not in self.predicted_ratings_df.index:
            return 0.0
        if movie_id not in self.predicted_ratings_df.columns:
            # Fallback to movie IMDb rating scaled if movie is in movies_df
            match = self.movies_df[self.movies_df["movie_id"] == movie_id]
            if not match.empty:
                return match["imdb_rating"].values[0] / 10.0
            return 0.0
            
        return self.predicted_ratings_df.at[user_id, movie_id]
        
    def recommend(self, user_id, top_n=10, exclude_watched=True):
        """
        Recommend top_n movies for a user using collaborative filtering predicted ratings.
        """
        if user_id not in self.predicted_ratings_df.index:
            # Cold start user: return popular movies as fallback
            popular = self.movies_df.sort_values(by=["imdb_rating"], ascending=False).head(top_n)
            return list(zip(popular["movie_id"].values, [0.5] * top_n))
            
        # Get user's predicted ratings
        user_preds = self.predicted_ratings_df.loc[user_id]
        
        scores = list(zip(user_preds.index, user_preds.values))
        
        if exclude_watched:
            # Get movies already watched by the user
            watched_movies = set(self.interactions_df[self.interactions_df["user_id"] == user_id]["movie_id"].values)
            scores = [(mid, score) for mid, score in scores if mid not in watched_movies]
            
        # Sort by predicted rating descending
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores[:top_n]

if __name__ == "__main__":
    cf = CollaborativeFilteringRecommender(num_factors=20)
    cf.fit()
    # Test recommendations
    test_user = "user_07271"
    recs = cf.recommend(test_user, top_n=5)
    print(f"\nPersonalized collaborative recommendations for {test_user}:")
    for mid, score in recs:
        title = cf.movies_df[cf.movies_df["movie_id"] == mid]["title"].values[0]
        print(f"  - {title} (ID: {mid}, predicted score: {score:.3f})")
