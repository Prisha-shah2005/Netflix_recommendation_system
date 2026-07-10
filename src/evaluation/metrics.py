"""
Netflix Recommendation System - Evaluation System
Author: Senior Machine Learning Engineer / Data Scientist

This script implements evaluation functions for measuring the performance of our recommendation models:
- Precision@K
- Recall@K
- MAP@K (Mean Average Precision)
- NDCG@K (Normalized Discounted Cumulative Gain)
- Coverage (catalog coverage)
- Diversity (intra-list similarity / distance)

It splits interactions into an 80% training set and 20% test set, fits SVD and Content-Based models on the training set,
runs prediction rankings for all test users, and computes standard evaluation metrics.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

# Add parent directory to path to allow importing models
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

class RecommenderEvaluator:
    def __init__(self, cleaned_dir=r"d:\project\Netflix Recommendation System\data\cleaned"):
        self.cleaned_dir = cleaned_dir
        self.interactions_df = pd.read_csv(os.path.join(cleaned_dir, "engineered_interactions.csv"))
        self.movies_df = pd.read_csv(os.path.join(cleaned_dir, "engineered_movie_features.csv"))
        
        # Prepare movie IDs and TF-IDF matrix for Diversity
        self.movie_ids = self.movies_df["movie_id"].values
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        tfidf_cols = [col for col in self.movies_df.columns if col.startswith("tfidf_")]
        self.tfidf_matrix = self.movies_df[tfidf_cols].values
        
        # Train / Test splits
        self.train_df = None
        self.test_df = None
        self.test_relevant_dict = {}  # user_id -> set of relevant movie_ids
        self.train_watched_dict = {}   # user_id -> set of watched movie_ids in training
        
    def split_data(self, test_size=0.2, random_state=42):
        """
        Splits interactions into train and test sets.
        Uses stratified user splitting: for each user, 20% of their interactions are placed in the test set.
        """
        print(f"Splits data into train/test with test_size={test_size}...")
        np.random.seed(random_state)
        
        train_rows = []
        test_rows = []
        
        # Group by user_id
        for user_id, group in self.interactions_df.groupby("user_id"):
            n_items = len(group)
            if n_items < 5:
                # Users with few interactions always go to the training set to prevent empty profiles
                train_rows.append(group)
            else:
                # Randomly split interactions for this user
                shuffled_indices = np.random.permutation(n_items)
                split_idx = int(n_items * (1 - test_size))
                
                train_idx = shuffled_indices[:split_idx]
                test_idx = shuffled_indices[split_idx:]
                
                train_rows.append(group.iloc[train_idx])
                test_rows.append(group.iloc[test_idx])
                
        self.train_df = pd.concat(train_rows).reset_index(drop=True)
        self.test_df = pd.concat(test_rows).reset_index(drop=True)
        
        # Map user's train-watched and test-relevant items
        # An item is relevant if progress is high (>= 50%) or user_rating is positive (like_dislike == 1)
        self.test_relevant_dict = {}
        for user_id, group in self.test_df.groupby("user_id"):
            relevant_movies = group[group["like_dislike"] >= 0]["movie_id"].values
            if len(relevant_movies) > 0:
                self.test_relevant_dict[user_id] = set(relevant_movies)
                
        self.train_watched_dict = {}
        for user_id, group in self.train_df.groupby("user_id"):
            self.train_watched_dict[user_id] = set(group["movie_id"].values)
            
        print(f"Data Split Stats: Train Interactions: {self.train_df.shape[0]}, Test Interactions: {self.test_df.shape[0]}")
        print(f"Evaluatable users in test set: {len(self.test_relevant_dict)}")

    def fit_eval_models(self):
        """
        Train Content-Based and SVD Collaborative filtering on the training interactions only.
        """
        print("Training models on training split...")
        
        # 1. Collaborative Filtering SVD on Train Matrix
        # Pivot train df
        train_matrix_df = self.train_df.pivot(index="user_id", columns="movie_id", values="interaction_score").fillna(0.0)
        
        # SVD Decomposition
        R = train_matrix_df.values
        user_ratings_mean = np.mean(R, axis=1)
        R_demeaned = R - user_ratings_mean.reshape(-1, 1)
        
        # Factorize with k latent vectors
        k = min(20, min(R_demeaned.shape) - 1)
        U, sigma, Vt = svds(R_demeaned, k=k)
        sigma = np.diag(sigma)
        
        all_pred = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        all_pred = np.clip(all_pred, 0.0, 1.2)
        
        self.cf_predictions = pd.DataFrame(all_pred, index=train_matrix_df.index, columns=train_matrix_df.columns)
        
        # 2. Content-Based Profiles from Train
        # User content profiles based on train
        self.user_content_profiles = {}
        for user_id, group in self.train_df.groupby("user_id"):
            movie_indices = []
            weights = []
            for _, row in group.iterrows():
                mid = row["movie_id"]
                if mid in self.movie_id_to_idx:
                    movie_indices.append(self.movie_id_to_idx[mid])
                    weights.append(row["interaction_score"])
            if movie_indices:
                user_vectors = self.tfidf_matrix[movie_indices]
                weights = np.array(weights).reshape(-1, 1)
                profile = np.sum(user_vectors * weights, axis=0)
                norm = np.linalg.norm(profile)
                if norm > 0:
                    profile = profile / norm
                self.user_content_profiles[user_id] = profile

    def _get_hybrid_recommendations(self, user_id, k=10, alpha=0.5):
        """
        Generates hybrid recommendations for a user based on evaluation model states.
        """
        # Fallback to popular if user not in train
        if user_id not in self.train_watched_dict:
            return self.movies_df.sort_values(by="imdb_rating", ascending=False)["movie_id"].head(k).tolist()
            
        # Get watched list to exclude
        watched = self.train_watched_dict.get(user_id, set())
        
        # Content scores
        c_profile = self.user_content_profiles.get(user_id, np.zeros(self.tfidf_matrix.shape[1]))
        if np.all(c_profile == 0.0):
            c_scores = np.zeros(len(self.movie_ids))
        else:
            c_scores = cosine_similarity(c_profile.reshape(1, -1), self.tfidf_matrix).flatten()
            
        # CF scores
        if user_id in self.cf_predictions.index:
            cf_preds = self.cf_predictions.loc[user_id]
            cf_scores = np.array([cf_preds.get(mid, 0.0) for mid in self.movie_ids])
        else:
            cf_scores = np.zeros(len(self.movie_ids))
            
        # Normalize and blend
        def min_max(arr):
            rng = arr.max() - arr.min()
            if rng == 0:
                return np.zeros_like(arr)
            return (arr - arr.min()) / rng
            
        norm_c = min_max(c_scores)
        norm_cf = min_max(cf_scores)
        
        # Dynamically shift weight to content based for users with sparse histories
        history_len = len(watched)
        if history_len < 5:
            user_alpha = 0.8
        else:
            user_alpha = alpha
            
        blended = user_alpha * norm_c + (1.0 - user_alpha) * norm_cf
        
        # Rank candidate movies
        scores = list(zip(self.movie_ids, blended))
        scores = [(mid, score) for mid, score in scores if mid not in watched]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        return [mid for mid, _ in scores[:k]]

    def evaluate(self, k=10):
        """
        Evaluates models and prints scores for Precision@K, Recall@K, MAP@K, NDCG@K, Coverage, and Diversity.
        """
        print(f"\nEvaluating Hybrid Recommender System at K={k}...")
        
        precisions = []
        recalls = []
        aps = []
        ndcgs = []
        
        all_recommended_items = set()
        user_list = list(self.test_relevant_dict.keys())
        
        # To compute diversity
        total_diversity = 0.0
        diversity_count = 0
        
        for user_id in user_list:
            relevant = self.test_relevant_dict[user_id]
            recs = self._get_hybrid_recommendations(user_id, k=k)
            
            # Record recommended items for coverage
            all_recommended_items.update(recs)
            
            # A. Precision & Recall @ K
            intersection = set(recs).intersection(relevant)
            precision = len(intersection) / k
            recall = len(intersection) / len(relevant) if len(relevant) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            
            # B. MAP (Mean Average Precision)
            # Calculate Average Precision (AP)
            ap = 0.0
            num_hits = 0
            for i, item in enumerate(recs):
                if item in relevant:
                    num_hits += 1
                    ap += num_hits / (i + 1)
            if len(relevant) > 0:
                ap /= len(relevant)
            aps.append(ap)
            
            # C. NDCG @ K
            # Discounted Cumulative Gain
            dcg = 0.0
            for i, item in enumerate(recs):
                if item in relevant:
                    dcg += 1.0 / np.log2(i + 2)
            # Ideal DCG (all active relevances at the top)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
            ndcg = dcg / idcg if idcg > 0.0 else 0.0
            ndcgs.append(ndcg)
            
            # D. Intra-List Diversity
            # Calculate average pairwise distance (1 - cosine similarity of TF-IDF vectors)
            if len(recs) > 1:
                rec_indices = [self.movie_id_to_idx[mid] for mid in recs if mid in self.movie_id_to_idx]
                if len(rec_indices) > 1:
                    rec_vectors = self.tfidf_matrix[rec_indices]
                    # Compute cosine similarity matrix
                    sim_matrix = cosine_similarity(rec_vectors)
                    # Average pairwise distance (upper triangle excluding diagonal)
                    n_rec = len(rec_indices)
                    pairwise_distances = []
                    for i in range(n_rec):
                        for j in range(i + 1, n_rec):
                            pairwise_distances.append(1.0 - sim_matrix[i, j])
                    if pairwise_distances:
                        total_diversity += np.mean(pairwise_distances)
                        diversity_count += 1
                        
        # Aggregate scores
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_map = np.mean(aps)
        mean_ndcg = np.mean(ndcgs)
        coverage = len(all_recommended_items) / len(self.movie_ids) * 100.0
        avg_diversity = (total_diversity / diversity_count) if diversity_count > 0 else 0.0
        
        print("\n==================================================")
        print("RECOMMENDATION METRICS SUMMARY")
        print("==================================================")
        print(f"Precision@{k}: {mean_precision:.4f}")
        print(f"  - Explanation: Percentage of recommendations that were liked by the user in testing.")
        print(f"Recall@{k}:    {mean_recall:.4f}")
        print(f"  - Explanation: Percentage of user's liked test items that were successfully recommended.")
        print(f"MAP@{k}:       {mean_map:.4f}")
        print(f"  - Explanation: Mean Average Precision. Penalizes relevant recommendations ranked low.")
        print(f"NDCG@{k}:      {mean_ndcg:.4f}")
        print(f"  - Explanation: Normalized Discounted Cumulative Gain. Considers ranking position utility.")
        print(f"Coverage:    {coverage:.2f}%")
        print(f"  - Explanation: Percentage of the catalog ({len(self.movie_ids)} items) recommended at least once.")
        print(f"Diversity:   {avg_diversity:.4f}")
        print(f"  - Explanation: Average distance (1 - cosine metadata similarity) between recommended movies.")
        print("==================================================")

if __name__ == "__main__":
    evaluator = RecommenderEvaluator()
    evaluator.split_data()
    evaluator.fit_eval_models()
    evaluator.evaluate(k=10)
