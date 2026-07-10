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
        self.interaction_matrix_df = pd.read_csv(self.matrix_path, index_col="user_id")
        self.interactions_df = pd.read_csv(self.interactions_path)
        self.movies_df = pd.read_csv(self.movies_path)
        
        self.user_ids = self.interaction_matrix_df.index.values
        self.movie_ids = self.interaction_matrix_df.columns.values
        
        # Convert to numpy matrix
        R = self.interaction_matrix_df.values
        
        # Calculate user rating means (only for non-zero interactions to avoid biasing towards zero)
        # Note: A simple standard way in SVD is to compute the mean of each user's active ratings
        user_ratings_mean = np.zeros(R.shape[0])
        for i in range(R.shape[0]):
            user_row = R[i, :]
            active_ratings = user_row[user_row > 0.0]
            if len(active_ratings) > 0:
                user_ratings_mean[i] = np.mean(active_ratings)
            else:
                user_ratings_mean[i] = 0.0
                
        # De-mean the rating matrix (subtract user mean from all non-zero ratings)
        R_demeaned = np.zeros(R.shape)
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if R[i, j] > 0.0:
                    R_demeaned[i, j] = R[i, j] - user_ratings_mean[i]
                    
        # Perform SVD (Singular Value Decomposition)
        # We find the top k (num_factors) singular values
        # If k is larger than or equal to matrix dimensions, handle it
        k = min(self.num_factors, min(R_demeaned.shape) - 1)
        U, sigma, Vt = svds(R_demeaned, k=k)
        
        # Convert sigma to a diagonal matrix
        sigma = np.diag(sigma)
        
        # Reconstruct the ratings matrix and add user means back
        all_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        
        # Add user rating means back
        for i in range(R.shape[0]):
            all_predicted_ratings[i, :] = all_predicted_ratings[i, :] + user_ratings_mean[i]
            
        # Clip predicted ratings between [0, 1.2] to match interaction score range
        all_predicted_ratings = np.clip(all_predicted_ratings, 0.0, 1.2)
        
        # Convert to DataFrame
        self.predicted_ratings_df = pd.DataFrame(
            all_predicted_ratings, 
            index=self.user_ids, 
            columns=self.movie_ids
        )
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
