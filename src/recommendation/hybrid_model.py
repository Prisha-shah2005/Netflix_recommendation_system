"""
Netflix Recommendation System - Hybrid Recommendation Coordinator
Author: Senior Machine Learning Engineer / Data Scientist

This script implements the HybridRecommender class which:
1. Instantiates and fits both Content-Based and Collaborative Filtering models.
2. Blends their scores using a weighted average:
   Score = alpha * Content_Score + (1 - alpha) * Collaborative_Score
3. Scales both score distributions to [0, 1] to ensure mathematical consistency.
4. Implements a dynamic alpha-weighting mechanism:
   - For users with a rich watch history (>= 5 sessions), alpha is set to 0.5 (balanced).
   - For cold-start/low-activity users (< 5 sessions), alpha is dynamically shifted up to 0.8
     (favoring Content-Based filtering to mitigate matrix sparsity issues).
   - For users with 0 sessions, recommends popular movies matching their search genres.
"""

import os
import pandas as pd
import numpy as np
from src.recommendation.content_based import ContentBasedRecommender
from src.recommendation.collaborative_filtering import CollaborativeFilteringRecommender

class HybridRecommender:
    def __init__(self, cleaned_dir=r"d:\project\Netflix Recommendation System\data\cleaned", num_factors=20, default_alpha=0.5):
        self.cleaned_dir = cleaned_dir
        self.default_alpha = default_alpha
        
        # Instantiate sub-models
        self.content_model = ContentBasedRecommender(cleaned_dir=cleaned_dir)
        self.collaborative_model = CollaborativeFilteringRecommender(cleaned_dir=cleaned_dir, num_factors=num_factors)
        
        self.movies_df = None
        self.interactions_df = None
        self.user_features_df = None
        
    def fit(self):
        """
        Fits both underlying models and loads reference data.
        """
        print("Fitting Hybrid Recommendation System...")
        self.content_model.fit()
        self.collaborative_model.fit()
        
        self.movies_df = pd.read_csv(os.path.join(self.cleaned_dir, "movies.csv"))
        self.interactions_df = pd.read_csv(os.path.join(self.cleaned_dir, "engineered_interactions.csv"))
        self.user_features_df = pd.read_csv(os.path.join(self.cleaned_dir, "engineered_user_features.csv"))
        print("Hybrid Recommendation System successfully fitted and ready.")
        
    def recommend(self, user_id, top_n=10, exclude_watched=True):
        """
        Generates personalized hybrid recommendations for the user.
        """
        # Determine user watch history richness
        user_history = self.interactions_df[self.interactions_df["user_id"] == user_id]
        history_count = len(user_history)
        
        # Dynamic alpha calculation:
        # If user has zero history, we default completely to content-based/popularity (alpha = 1.0)
        # If user has low history (1-4 items), we favor content-based (alpha = 0.8) to handle cold-start
        # If user has rich history (>= 5 items), we balance both (alpha = default_alpha, e.g., 0.5)
        if history_count == 0:
            alpha = 1.0
        elif history_count < 5:
            alpha = 0.8
        else:
            alpha = self.default_alpha
            
        # Get content-based scores for all catalog items (not capped)
        content_recs = self.content_model.recommend(user_id, top_n=1000, exclude_watched=exclude_watched)
        content_dict = {mid: score for mid, score in content_recs}
        
        # Get collaborative filtering scores for all catalog items
        collab_recs = self.collaborative_model.recommend(user_id, top_n=1000, exclude_watched=exclude_watched)
        collab_dict = {mid: score for mid, score in collab_recs}
        
        # Merge candidate keys
        candidate_ids = set(content_dict.keys()).union(set(collab_dict.keys()))
        
        # Norm function to scale scores to [0, 1] range safely
        def scale_scores(score_dict):
            if not score_dict:
                return {}
            vals = np.array(list(score_dict.values()))
            min_val, max_val = vals.min(), vals.max()
            diff = max_val - min_val
            if diff == 0:
                return {k: 0.5 for k in score_dict.keys()}
            return {k: float((v - min_val) / diff) for k, v in score_dict.items()}
            
        scaled_content = scale_scores(content_dict)
        scaled_collab = scale_scores(collab_dict)
        
        # Blend scores
        hybrid_scores = []
        for mid in candidate_ids:
            c_score = scaled_content.get(mid, 0.0)
            cf_score = scaled_collab.get(mid, 0.0)
            
            # Weighted average formula
            blended_score = alpha * c_score + (1.0 - alpha) * cf_score
            hybrid_scores.append((mid, blended_score))
            
        # Sort by blended score descending
        hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
        return hybrid_scores[:top_n]
        
    def get_movie_details(self, movie_id):
        """
        Helper method to fetch full details of a movie for presentation.
        """
        match = self.movies_df[self.movies_df["movie_id"] == movie_id]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

if __name__ == "__main__":
    hr = HybridRecommender()
    hr.fit()
    # Test recommendations
    test_user = "user_07271"
    recs = hr.recommend(test_user, top_n=5)
    print(f"\nPersonalized Hybrid recommendations for {test_user}:")
    for mid, score in recs:
        details = hr.get_movie_details(mid)
        print(f"  - {details['title']} (ID: {mid}, Blend Score: {score:.3f}, Genre: {details['genre_primary']}, Lang: {details['language']})")
