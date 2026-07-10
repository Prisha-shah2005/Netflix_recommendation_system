"""
Netflix Recommendation System - Feature Engineering
Author: Senior Machine Learning Engineer / Data Scientist

This script implements the FeatureEngineer class which:
1. Loads the cleaned datasets.
2. Builds movie content metadata representations and extracts features using TF-IDF.
3. Calculates user preference features (genre weights, language weights, watch frequency, search frequency).
4. Computes a standardized User-Movie Interaction Score based on watch history, ratings, and search interactions.
5. Constructs the User-Movie Interaction Matrix.
6. Serializes the generated feature objects for model training.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureEngineer:
    def __init__(self, cleaned_dir=None):
        if cleaned_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cleaned_dir = os.path.join(base_dir, "data", "cleaned")
        self.cleaned_dir = cleaned_dir
        self.movies_path = os.path.join(cleaned_dir, "movies.csv")
        self.watch_path = os.path.join(cleaned_dir, "watch_history.csv")
        self.search_path = os.path.join(cleaned_dir, "search_logs.csv")
        self.reviews_path = os.path.join(cleaned_dir, "reviews.csv")
        
        # Loaded dataframes
        self.movies_df = None
        self.watch_df = None
        self.search_df = None
        self.reviews_df = None
        
        # Engineered outputs
        self.movie_features = None  # DataFrame of movie IDs and metadata vectors
        self.user_features = None   # DataFrame of user IDs and stats/preference distributions
        self.interaction_df = None   # Flat DataFrame of user_id, movie_id, interaction_score, like_dislike
        self.tfidf_vectorizer = None
        
    def load_data(self):
        print("Loading cleaned datasets for feature engineering...")
        self.movies_df = pd.read_csv(self.movies_path)
        self.watch_df = pd.read_csv(self.watch_path)
        self.search_df = pd.read_csv(self.search_path)
        self.reviews_df = pd.read_csv(self.reviews_path)
        print(f"Loaded Movies: {self.movies_df.shape}, Watch Logs: {self.watch_df.shape}, Search Logs: {self.search_df.shape}, Reviews: {self.reviews_df.shape}")
        
    def engineer_movie_content_features(self):
        """
        Build text descriptions for each movie using metadata (title, content_type, genres, rating, language, country, is_netflix_original)
        and fit a TF-IDF model to represent movies as feature vectors.
        """
        print("Building Movie Metadata TF-IDF Profiles...")
        
        # Combine relevant textual attributes into a single text metadata string
        def create_soup(row):
            netflix_status = "netflix_original" if row["is_netflix_original"] else ""
            warning_status = "content_warning" if row["content_warning"] else ""
            soup = f"{row['title']} {row['content_type']} {row['genre_primary']} {row['genre_secondary']} " \
                   f"{row['rating']} {row['language']} {row['country_of_origin']} {netflix_status} {warning_status}"
            return soup
            
        self.movies_df["metadata_soup"] = self.movies_df.apply(create_soup, axis=1)
        
        # Initialize and fit TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df["metadata_soup"])
        
        # Convert to DataFrame
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
        tfidf_df["movie_id"] = self.movies_df["movie_id"]
        
        # Merge back with key metadata columns for model references
        self.movie_features = self.movies_df[["movie_id", "title", "content_type", "genre_primary", "genre_secondary", "language", "imdb_rating"]].merge(tfidf_df, on="movie_id")
        print(f"Movie features created. Shape: {self.movie_features.shape}")
        
    def engineer_user_profiles(self):
        """
        Calculate user behavior features:
        - Genre preferences (normalized proportion of watched movie genres)
        - Language preferences (normalized proportion of watched movie languages)
        - Watch frequency (total watch sessions)
        - Search frequency (total search logs count)
        - Recently watched (list of movie IDs)
        """
        print("Computing User Preference Profiles & Frequencies...")
        
        # Join watch history with movie details
        watch_movie = self.watch_df.merge(self.movies_df[["movie_id", "genre_primary", "language"]], on="movie_id", how="inner")
        
        # 1. Watch & Search frequencies
        watch_counts = self.watch_df.groupby("user_id").size().rename("watch_frequency")
        search_counts = self.search_df.groupby("user_id").size().rename("search_frequency")
        
        # 2. Favorite genre & genre preference distributions
        user_genre_counts = watch_movie.groupby(["user_id", "genre_primary"]).size().unstack(fill_value=0)
        user_genre_pref = user_genre_counts.div(user_genre_counts.sum(axis=1), axis=0) # Normalize to sum to 1
        user_genre_pref.columns = [f"genre_pref_{col.lower().replace(' ', '_')}" for col in user_genre_pref.columns]
        
        # 3. Language preference distributions
        user_lang_counts = watch_movie.groupby(["user_id", "language"]).size().unstack(fill_value=0)
        user_lang_pref = user_lang_counts.div(user_lang_counts.sum(axis=1), axis=0) # Normalize to sum to 1
        user_lang_pref.columns = [f"lang_pref_{col.lower().replace(' ', '_')}" for col in user_lang_pref.columns]
        
        # 4. Favorite content types
        user_type_counts = watch_movie.groupby(["user_id", "movie_id"]).size().reset_index()
        # Merge with content_type
        user_type_counts = user_type_counts.merge(self.movies_df[["movie_id", "content_type"]], on="movie_id")
        user_type_counts = user_type_counts.groupby(["user_id", "content_type"]).size().unstack(fill_value=0)
        user_type_pref = user_type_counts.div(user_type_counts.sum(axis=1), axis=0)
        user_type_pref.columns = [f"type_pref_{col.lower().replace(' ', '_')}" for col in user_type_pref.columns]

        # Combine all user features
        user_list = pd.DataFrame(index=self.watch_df["user_id"].unique())
        user_list.index.name = "user_id"
        
        self.user_features = user_list.join(watch_counts, how="left").join(search_counts, how="left")
        self.user_features["watch_frequency"] = self.user_features["watch_frequency"].fillna(0).astype(int)
        self.user_features["search_frequency"] = self.user_features["search_frequency"].fillna(0).astype(int)
        
        self.user_features = self.user_features.join(user_genre_pref, how="left").join(user_lang_pref, how="left").join(user_type_pref, how="left")
        
        # Fill any missing preferences for users with zero watch history with neutral (1 / count) values
        self.user_features = self.user_features.fillna(0.0)
        self.user_features = self.user_features.reset_index()
        print(f"User features created. Shape: {self.user_features.shape}")
        
    def calculate_interaction_scores(self):
        """
        Compute standard User-Movie Interaction Scores:
        - Watch Score (0.5 weight): based on progress_percentage. If >=70% or action is 'completed', score = 1.0.
        - Rating Score (0.4 weight): based on explicit user_rating in watch history, or rating/sentiment in reviews.
          - Ratings (1 to 5) mapped to [0, 1]
          - Sentiment score (0.0 to 1.0)
        - Search Score (0.1 weight): boost of 1.0 if the user searched for the movie's primary genre or titles.
        
        Additionally, creates:
        - `like_dislike`: Explicit rating >=4 or positive sentiment -> 1, rating <=2 or negative sentiment -> -1, else 0.
        """
        print("Calculating User-Movie Interaction Scores & Like/Dislike Flags...")
        
        # Create a unique set of user-movie pairs from watch history and reviews
        watch_pairs = self.watch_df[["user_id", "movie_id", "progress_percentage", "user_rating", "action"]].copy()
        review_pairs = self.reviews_df[["user_id", "movie_id", "rating", "sentiment_score", "sentiment"]].copy()
        
        # Deduplicate pairs to get base dataframe
        # Group watch history pairs by user and movie to get max progress and rating
        watch_grouped = watch_pairs.groupby(["user_id", "movie_id"]).agg({
            "progress_percentage": "max",
            "user_rating": "max",
            "action": lambda x: "completed" if "completed" in x.values else "started"
        }).reset_index()
        
        # Group reviews to get max rating and sentiment
        review_grouped = review_pairs.groupby(["user_id", "movie_id"]).agg({
            "rating": "max",
            "sentiment_score": "max",
            "sentiment": "first"
        }).reset_index()
        
        # Merge watch history and reviews
        interaction = pd.merge(watch_grouped, review_grouped, on=["user_id", "movie_id"], how="outer")
        
        # Fetch movie genre details to construct Search Score
        interaction = interaction.merge(self.movies_df[["movie_id", "genre_primary", "title"]], on="movie_id", how="left")
        
        # A. Compute Watch Score (implicit)
        # progress_percentage / 100
        interaction["watch_score"] = interaction["progress_percentage"].fillna(0.0) / 100.0
        # If marked completed, make it 1.0
        interaction.loc[interaction["action"] == "completed", "watch_score"] = 1.0
        
        # B. Compute Rating Score (explicit)
        # We blend rating (1-5 scaled to 0-1) and review rating, falling back to sentiment_score
        interaction["explicit_rating"] = interaction["user_rating"].fillna(interaction["rating"])
        
        # Scale explicit ratings from [1, 5] to [0, 1]
        interaction["rating_score"] = (interaction["explicit_rating"] - 1.0) / 4.0
        # For entries without explicit ratings, use reviews sentiment score
        interaction["rating_score"] = interaction["rating_score"].fillna(interaction["sentiment_score"])
        # Fallback for remaining is 0.0 (no rating signal)
        interaction["rating_score"] = interaction["rating_score"].fillna(0.0)
        
        # C. Compute Search Score (implicit interest)
        # Check if the user searched for the movie's genre or title before
        # Pre-cache user searches to speed up checking
        user_searches = self.search_df.groupby("user_id")["search_query"].apply(lambda x: " ".join(x.values)).to_dict()
        
        def calculate_search_signal(row):
            user_id = row["user_id"]
            if user_id not in user_searches:
                return 0.0
            queries = user_searches[user_id]
            genre = str(row["genre_primary"]).lower()
            title = str(row["title"]).lower()
            
            # Simple keyword matching: does query contain genre or parts of title
            if genre in queries:
                return 1.0
            # Check title words (excluding short words)
            for word in title.split():
                if len(word) > 3 and word in queries:
                    return 1.0
            return 0.0
            
        interaction["search_score"] = interaction.apply(calculate_search_signal, axis=1)
        
        # Compute Blended Interaction Score
        # weights: progress=0.5, rating/sentiment=0.4, search=0.1
        interaction["interaction_score"] = (
            0.5 * interaction["watch_score"] +
            0.4 * interaction["rating_score"] +
            0.1 * interaction["search_score"]
        )
        
        # D. Compute Like / Dislike Class Label
        # 1 = Like, -1 = Dislike, 0 = Neutral
        # Criteria:
        # - Explicit rating >= 4 or positive review sentiment -> Like (1)
        # - Explicit rating <= 2 or negative review sentiment -> Dislike (-1)
        # - Else Neutral (0)
        interaction["like_dislike"] = 0
        
        like_mask = (interaction["explicit_rating"] >= 4.0) | (interaction["sentiment"] == "positive")
        dislike_mask = (interaction["explicit_rating"] <= 2.0) | (interaction["sentiment"] == "negative")
        
        interaction.loc[like_mask, "like_dislike"] = 1
        interaction.loc[dislike_mask, "like_dislike"] = -1
        
        # Drop temporary columns and clean output
        self.interaction_df = interaction[[
            "user_id", "movie_id", "interaction_score", "like_dislike", 
            "progress_percentage", "explicit_rating", "sentiment"
        ]].copy()
        
        print(f"User-Movie Interaction Scores computed. Total interactions: {self.interaction_df.shape[0]}")
        print(f"Like Count (1): {(self.interaction_df['like_dislike'] == 1).sum()}")
        print(f"Dislike Count (-1): {(self.interaction_df['like_dislike'] == -1).sum()}")
        print(f"Neutral Count (0): {(self.interaction_df['like_dislike'] == 0).sum()}")
        
    def build_interaction_matrix(self):
        """
        Pivot interaction DataFrame into a sparse User-Movie Interaction Matrix.
        Missing interactions represent unobserved ratings (filled with 0.0).
        """
        print("Constructing User-Movie Interaction Matrix...")
        interaction_matrix = self.interaction_df.pivot(
            index="user_id", 
            columns="movie_id", 
            values="interaction_score"
        ).fillna(0.0)
        print(f"Interaction Matrix Shape: {interaction_matrix.shape}")
        return interaction_matrix
        
    def save_features(self):
        """
        Saves all engineered feature dataframes to CSV inside data/cleaned/
        """
        print("Saving engineered features to data/cleaned/...")
        
        self.movie_features.to_csv(os.path.join(self.cleaned_dir, "engineered_movie_features.csv"), index=False)
        self.user_features.to_csv(os.path.join(self.cleaned_dir, "engineered_user_features.csv"), index=False)
        self.interaction_df.to_csv(os.path.join(self.cleaned_dir, "engineered_interactions.csv"), index=False)
        
        # Build and save pivot matrix
        matrix = self.build_interaction_matrix()
        matrix.to_csv(os.path.join(self.cleaned_dir, "interaction_matrix.csv"))
        
        print("Feature engineering successfully completed and saved!")

if __name__ == "__main__":
    fe = FeatureEngineer()
    fe.load_data()
    fe.engineer_movie_content_features()
    fe.engineer_user_profiles()
    fe.calculate_interaction_scores()
    fe.save_features()
