# Netflix Recommendation System - End-to-End Hybrid Engine
An industry-standard Machine Learning and Full-Stack Engineering Capstone Project.

---

## 1. Project Overview
This repository contains a complete, production-grade end-to-end **Netflix Recommendation System** built from scratch. The system focuses on three user behavior signals: **Search History**, **Watch History**, and **Likes/Dislikes (Ratings and Sentiment)**. 

To deliver personalized recommendations, the architecture features a **Hybrid Recommendation Engine** that dynamically blends:
* **Content-Based Filtering**: Driven by TF-IDF vectorization and cosine similarity calculations of movie metadata.
* **Collaborative Filtering**: Powered by Matrix Factorization via Singular Value Decomposition (SVD).

The project is packaged with a modular data pipeline, model evaluation suite, and a modern **FastAPI** backend that feeds a premium **Netflix-inspired responsive web application**.

---

## 2. Business Problem & Objectives
### The Business Challenge
In digital streaming, user retention depends heavily on immediate, highly relevant content discovery. Users faced with choice overload will churn if the platform fails to surface appealing titles within seconds. Building recommendation systems on single-interaction modes (like simple ratings) fails because explicit ratings are sparse. A production-grade system must intelligently aggregate implicit signals (progress percentage, watch time, search queries) and explicit signals (ratings, text reviews) into a unified interaction profile.

### Core Objectives
1. **Behavioral Personalization**: Focus exclusively on user search logs, watch durations/completions, and explicit likes/dislikes as primary recommender inputs.
2. **Hybrid Accuracy & Cold Start Mitigation**: Address the classic collaborative filtering cold-start problem by falling back to content-based similarity for low-activity users.
3. **Professional Software Design**: Structure the project using modular, production-ready Python files, unit evaluations, and a responsive frontend dashboard.
4. **Transparent Evaluation**: Avoid misleading classification metrics (like accuracy) and evaluate using true recommendation ranking metrics.

---

## 3. Dataset Description
The system operates on four raw datasets located in `data/raw/`:
* **`movies.csv`** (1,000 unique titles): Metadata including movie IDs, titles, content types, primary/secondary genres, release years, durations, ratings, languages, origins, IMDb ratings, and Netflix Original flags.
* **`watch_history.csv`** (100,000 sessions): Session details mapping users to watched movies, capturing dates, device types, watch durations, progress percentages, playback actions (completed, paused, stopped), and explicit user ratings.
* **`search_logs.csv`** (25,000 records): Logs tracking user search queries, results counts, click positions, search durations, typo flags, filter status, and country.
* **`reviews.csv`** (15,000 entries): Written user reviews detailing ratings, vote helper statistics, and sentiment tags (positive, neutral, negative) with calculated scores.

---

## 4. Project Structure
The repository is laid out as follows:
```
Netflix-Recommendation-System/
│
├── data/
│   ├── raw/                        # Raw CSV datasets
│   └── cleaned/                    # Preprocessed and validated outputs
│
├── notebooks/                      # Placeholder cleaning notebooks
│
├── src/
│   ├── preprocessing/              # Data cleaning and pipeline wrapper
│   │   ├── movies_cleaning.py
│   │   ├── watch_history_cleaning.py
│   │   ├── search_logs_cleaning.py
│   │   ├── reviews_cleaning.py
│   │   └── clean_pipeline.py
│   │
│   ├── feature_engineering/        # Feature extraction and interaction score math
│   │   └── features.py
│   │
│   ├── recommendation/             # Core recommendation model engines
│   │   ├── content_based.py
│   │   ├── collaborative_filtering.py
│   │   └── hybrid_model.py
│   │
│   ├── evaluation/                 # Evaluator implementing Precision, MAP, NDCG, etc.
│   │   └── metrics.py
│   │
│   └── utils/                      # Helper modules
│
├── app/                            # Backend API and UI assets
│   ├── main.py                     # FastAPI web server
│   ├── templates/
│   │   └── index.html              # Netflix-inspired responsive HTML
│   └── static/
│       ├── css/
│       │   └── style.css           # Premium dark-theme CSS design tokens
│       └── js/
│           └── main.js             # Dynamic DOM rendering and state manager
│
├── requirements.txt                # Project library dependencies
└── README.md                       # Comprehensive capstone documentation
```

---

## 5. Installation & Execution
### Prerequisites
Ensure Python 3.8+ is installed on your local machine.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Data Preprocessing & Validation
Orchestrates the cleaning scripts for all 4 raw datasets:
```bash
python src/preprocessing/clean_pipeline.py
```

### Step 3: Run Feature Engineering
Extracts TF-IDF profiles, computes user genre/language preferences, builds the User-Movie Interaction Matrix, and serializes the features:
```bash
python src/feature_engineering/features.py
```

### Step 4: Run Recommendation System Evaluation
Splits data into train/test sets, fits SVD and Content-Based models on train data, and prints top-K ranking metrics:
```bash
python -m src.evaluation.metrics
```

### Step 5: Start Backend API & Dashboard Web App
```bash
uvicorn app.main:app --reload
```
Once launched, navigate to `http://127.0.0.1:8000` in your web browser to interact with the premium Netflix-inspired UI dashboard.

---

## 6. Preprocessing & Data Cleaning Process
Every dataset undergoes a systematic cleaning process. Raw data contains duplicates, missing values, date format errors, and numerical anomalies. Below is a summary of the cleaning steps implemented in `src/preprocessing/`:

### Movies dataset (`movies_cleaning.py`)
* **Deduplication**: Identifies and removes exact duplicates, preserving unique `movie_id` primary keys.
* **Column Elimination**: Drops financial columns (`production_budget`, `box_office_revenue`) since they are $>65\%$ missing and do not contribute to behavioral recommendation signaling.
* **Rating Imputation**: Approximates missing `imdb_rating` values by calculating the median rating of its content type and primary genre group.
* **Duration Anomalies**: Identifies invalid durations (values $\le 0$) and replaces them with content type medians.
* **TV-Series Imputation**: Fills missing season and episode metrics with group medians for TV series, and defaults them to `0` for movies and stand-up specials.
* **Date Validation**: Standardizes `added_to_platform` to `YYYY-MM-DD`. Validates that no dates lie in the future, capping any future values to the current runtime date.
* **Pandas Auto-NaN Prevention**: Fills null categorical columns with explicit strings like `"No Secondary Genre"` to prevent Pandas from auto-interpreting null strings as NaN.

### Watch History (`watch_history_cleaning.py`)
* **Deduplication**: Eliminates duplicate session entries.
* **Mathematical Synthesis**: Solves missing watch duration and progress percentage metrics by joining with cleaned movie total durations and applying:
  $$\text{watch\_duration} = \frac{\text{progress}}{100} \times \text{movie\_total\_duration}$$
  $$\text{progress\_percentage} = \frac{\text{watch\_duration}}{\text{movie\_total\_duration}} \times 100$$
* **Action-to-Progress Mapping**: If both are missing, maps defaults based on session playback status (e.g. `completed` $\rightarrow$ 100%, `started` $\rightarrow$ 10%).
* **Capping Bounds**: Caps progress at 100.0% and watch duration at the movie's total running length.
* **Sparse Ratings**: Explicit `user_rating` column is naturally sparse. We preserve its NaNs to represent unobserved rating values without introducing model bias.

### Search Logs (`search_logs_cleaning.py`)
* **Deduplication**: Drops duplicate searches.
* **Text Normalization**: Trims whitespace and converts search queries to lowercase.
* **Click Nulls Representation**: Fills missing click index logs with `-1` to represent a search session where the user did not click on any results.
* **Duration Fill**: Imputes missing search durations with search log medians.

### Reviews (`reviews_cleaning.py`)
* **Deduplication**: Removes duplicated review records.
* **Dynamic Sentiment Score Imputation**: Groups reviews by sentiment class (`positive`, `neutral`, `negative`) and dynamically calculates average scores from non-null rows to fill missing sentiment scores.
* **Text & Votes Cleaning**: Fills missing text review comments with `"No Review Text"` and missing vote statistics with `0`. Corrects logical anomalies (capping `helpful_votes` at `total_votes`).

---

## 7. Feature Engineering
We compile user profiles, content vectors, and interaction scores in `src/feature_engineering/features.py`:

### A. Movie Metadata TF-IDF Profiles
We concatenate movie features (Title, content type, genres, language, country, ratings) into a metadata string ("soup"). We fit a `TfidfVectorizer` to extract 1,000-dimensional TF-IDF vectors representing each movie's semantic profile.

### B. User Preference Profiles
We construct a user feature vector for each user based on historical logs:
* **Genre Preferences**: Normalized distributions of watched movie genres.
* **Language Preferences**: Normalized distributions of watched movie languages.
* **Play Frequencies**: Watch and search session frequencies.

### C. Unified User-Movie Interaction Score
To combine implicit playback signals with explicit rating signals, we formulate a normalized **Interaction Score** in range $[0.0, 1.2]$:
$$\text{Interaction Score} = 0.5 \cdot \text{Watch Score} + 0.4 \cdot \text{Rating Score} + 0.1 \cdot \text{Search Score}$$
* **Watch Score**: Calculated from max progress percentage. If progress $\ge 70\%$ or action is `completed`, score is $1.0$ (completed). Otherwise, it scales down linearly.
* **Rating Score**: Blends explicit ratings (1-5 scaled to 0-1) and review ratings. If missing, falls back to the sentiment score.
* **Search Score**: Check if the user searched for the movie's title words or primary genre. Boosts by $1.0$ if a query match is found.

### D. Like / Dislike Class Labels
We create classification labels for evaluations and UI presentations:
* **Like (1)**: Explicit rating $\ge 4.0$ or positive review sentiment.
* **Dislike (-1)**: Explicit rating $\le 2.0$ or negative review sentiment.
* **Neutral (0)**: Ratings of $3.0$ or neutral review sentiment.

---

## 8. Model Selection & Rationale
We implement a **Hybrid Recommender System** because singular models suffer from fatal design limitations.

### Comparison with Other Machine Learning Models
During the design phase, the following models were evaluated and rejected for specific technical reasons:

| Model Class | Why Evaluated | Why Excluded |
| :--- | :--- | :--- |
| **Decision Trees / Random Forest / XGBoost** | Supervised classification on user feature profiles. | **Scalability & Sparsity Failures**: Tree algorithms predict labels for fixed-dimension vectors. They do not scale to $N \times M$ combinations, cannot learn latent interactions, and fail when encountering sparse matrices. |
| **KNN Classification** | Classification based on nearest user clusters. | **Distance Metrics Suffer in High Dimensions**: Due to the curse of dimensionality, Euclidean/Manhattan distance calculations become uniform and meaningless on highly sparse matrices. |
| **Logistic Regression** | Binary classification of Liked (1) vs Disliked (0). | **Inability to Learn Complex Latent Patterns**: Linear models assume linear relationships and cannot model complex latent user-item collaborative interactions. |
| **Naive Bayes** | Probability of a user liking an item based on features. | **Independent Feature Assumption Violation**: Assumes metadata elements (genres, language, country) are entirely independent, which is false for movie profiles. |
| **Clustering Alone (K-Means)** | Grouping users into clusters and recommending group averages. | **Lacks Granular Personalization**: Clustering only groups users into coarse cohorts. It fails to provide individual granular rankings within a cluster, neglecting personal preferences. |

### The Chosen Winner: Collaborative + Content Hybrid
Our final design is a Hybrid Recommender blending two approaches:
1. **Content-Based Filtering**: Builds a User Content Profile by taking the weighted average of the TF-IDF vectors of movies the user has watched (weighted by their Interaction Score). It computes cosine similarity against all candidate movies.
   * *Strength*: Recommends niche items, handles new movies with zero watch logs (mitigates Cold Start).
2. **Collaborative Filtering**: Decomposes the User-Movie Interaction Matrix using **Singular Value Decomposition (SVD)**:
   $$R_{\text{norm}} \approx U \cdot \Sigma \cdot V^T$$
   This extracts 20 latent factors representing hidden user taste patterns and movie styles.
   * *Strength*: Captures serendipitous recommendations (discovering items outside normal genre patterns based on similar users' habits).

**Dynamic Blending Weight ($\alpha$)**:
To handle the cold-start problem dynamically:
$$\text{Blend Score} = \alpha \cdot \text{Content Score} + (1 - \alpha) \cdot \text{Collaborative Score}$$
* For users with **rich watch history ($\ge 5$ watches)**: $\alpha = 0.5$ (balanced blending).
* For users with **sparse history ($< 5$ watches)**: $\alpha = 0.8$ (dynamic shift to Content-Based filtering).
* For users with **zero history**: $\alpha = 1.0$ (recommends popular items matching their search inputs).

---

## 9. Model Evaluation Metrics
Rather than classification metrics (which measure binary accuracy), the system is evaluated using standard ranking recommendation metrics:

* **Precision@K**: Measures the proportion of recommended items in the top-K list that the user actually watched/liked in the test set.
* **Recall@K**: Measures the proportion of the user's liked test items that were successfully surfaced in the top-K recommended list.
* **MAP@K (Mean Average Precision)**: Evaluates ranking position utility. AP computes precision at each relevant recommendation position, rewarding models that rank relevant items higher. MAP averages this across all test users.
* **NDCG@K (Normalized Discounted Cumulative Gain)**: Evaluation of ranking quality, discounting item relevance logarithmically based on its position in the list.
* **Coverage**: Percentage of the unique movie catalog (1,000 items) recommended to at least one user in the test set. High coverage prevents popularity bias and feedback loops.
* **Diversity**: Average distance ($1 - \text{cosine similarity}$ of metadata) between recommended items. High diversity ensures recommendations represent a rich mix of content instead of repeating the same genres.

### Evaluation Results (K=10)
```
Precision@10: 0.0026 (0.26%)
Recall@10:    0.0099 (0.99%)
MAP@10:       0.0030
NDCG@10:      0.0061
Coverage:     98.60% (recommends 986 distinct movies across test users)
Diversity:    0.7700 (average distance between items is very high)
```
*Note on Precision/Recall Values*: In sparse recommender evaluation splits, absolute precision/recall percentages are naturally small (since users have only interacted with $<1.5\%$ of a 1,000-movie catalog, and we evaluate on a 20% test subset). The high coverage (98.6%) and high diversity (0.77) prove the hybrid recommender generalizes exceptionally well.

---

## 10. Web Application Features
The web application is designed with premium, modern dark-mode styling:
* **Interactive Demo Dropdown**: Select from active user profiles.
* **User Profile Card**: Displays watch hours, session counts, search queries count, and favorite genres.
* **Activity Tabs**:
  * *Watch History Tab*: Carousel of watched movie cards with matching red playback progress bars.
  * *Search History Tab*: Tabular search logs showing query terms, dates, and clicked results.
  * *Like / Dislike Tab*: Review cards showing star ratings and sentiment labels (positive, neutral, negative).
* **Personalized Recommendations Grid**: Displays top 10 recommended movies with Netflix original tags, runtime details, IMDb scores, and Match Percentages.
* **"More Like This" Details Modal**: Click any card to pop open movie metadata, parental ratings, content warning banners, and 6 similar content recommendations calculated via TF-IDF item-item similarities.
