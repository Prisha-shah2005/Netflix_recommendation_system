/**
 * Netflix Recommendation System - Interactive Frontend Coordinator
 * Author: Senior Machine Learning Engineer / Data Scientist
 * 
 * Manages UI state, API fetch requests, dynamic DOM rendering, 
 * tab switching, and detail modals.
 */

document.addEventListener("DOMContentLoaded", () => {
    // UI Elements
    const userSelect = document.getElementById("user-select");
    const welcomeHero = document.getElementById("welcome-hero");
    const profileInsights = document.getElementById("profile-insights");
    const recommendationSection = document.getElementById("recommendation-section");
    
    // Stats Elements
    const statHours = document.getElementById("stat-hours");
    const statWatches = document.getElementById("stat-watches");
    const statSearches = document.getElementById("stat-searches");
    const statRatings = document.getElementById("stat-ratings");
    const statGenre = document.getElementById("stat-genre");
    const activeUserBadge = document.getElementById("active-user-badge");
    
    // Grid/Table Container Elements
    const watchGrid = document.getElementById("watch-history-grid");
    const searchTableBody = document.getElementById("search-history-table-body");
    const ratingsGrid = document.getElementById("ratings-history-grid");
    const recsGrid = document.getElementById("recommendations-grid");
    
    // Tabs Elements
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");
    
    // Modal Elements
    const modal = document.getElementById("movie-detail-modal");
    const closeModalBtn = document.getElementById("close-modal");
    const modalBody = document.getElementById("modal-movie-body");
    
    // Initialize Dashboard: Load Demo Profiles
    fetchDemoUsers();
    setupTabSwitching();
    setupModalClosing();

    /**
     * Fetch user profiles to populate the select dropdown.
     */
    async function fetchDemoUsers() {
        try {
            const response = await fetch("/api/users");
            if (!response.ok) throw new Error("Failed to fetch demo users");
            
            const users = await response.json();
            
            // Clear loading placeholder
            userSelect.innerHTML = '<option value="" disabled selected>Select a Profile...</option>';
            
            users.forEach(user => {
                const opt = document.createElement("option");
                opt.value = user.user_id;
                opt.textContent = `${user.user_id} (${user.watch_frequency} watched | Fav: ${user.favorite_genre})`;
                userSelect.appendChild(opt);
            });
            
            // Event listener on profile select
            userSelect.addEventListener("change", (e) => {
                const userId = e.target.value;
                if (userId) {
                    loadUserProfile(userId);
                }
            });
            
        } catch (error) {
            console.error("Error loading demo users:", error);
            userSelect.innerHTML = '<option value="" disabled>Error loading profiles</option>';
        }
    }

    /**
     * Load the user profile statistics and histories, and fetch recommendations.
     */
    async function loadUserProfile(userId) {
        // Toggle view visibility
        welcomeHero.classList.add("hidden");
        profileInsights.classList.remove("hidden");
        recommendationSection.classList.remove("hidden");
        
        activeUserBadge.textContent = userId;
        
        // Show loaders
        watchGrid.innerHTML = '<div class="loader">Loading Watch History...</div>';
        searchTableBody.innerHTML = '<tr><td colspan="4" class="text-center">Loading Search History...</td></tr>';
        ratingsGrid.innerHTML = '<div class="loader">Loading Ratings...</div>';
        recsGrid.innerHTML = '<div class="loader">Generating Hybrid Recommendations...</div>';
        
        // Fetch histories
        try {
            const response = await fetch(`/api/user/${userId}/history`);
            if (!response.ok) throw new Error("Failed to load user history");
            const data = await response.json();
            
            // 1. Populate Stats Cards
            statHours.textContent = data.stats.total_watch_hours;
            statWatches.textContent = data.stats.watch_count;
            statSearches.textContent = data.stats.search_count;
            statRatings.textContent = data.stats.rating_count;
            statGenre.textContent = data.stats.favorite_genre;
            
            // 2. Render Watch History Tab
            renderWatchHistory(data.watch_history);
            
            // 3. Render Search History Tab
            renderSearchHistory(data.search_history);
            
            // 4. Render Likes/Dislikes Tab
            renderRatingsHistory(data.ratings_history);
            
        } catch (err) {
            console.error(err);
            // Show error state in tabs
            watchGrid.innerHTML = '<div class="error-msg">Error loading watch history.</div>';
        }
        
        // Fetch Recommendations
        try {
            const response = await fetch(`/api/user/${userId}/recommendations`);
            if (!response.ok) throw new Error("Failed to fetch recommendations");
            const recs = await response.json();
            
            renderRecommendations(recs);
        } catch (err) {
            console.error(err);
            recsGrid.innerHTML = '<div class="error-msg">Failed to retrieve recommendations. Please try again.</div>';
        }
    }

    /**
     * Render watch history card elements.
     */
    function renderWatchHistory(historyList) {
        if (!historyList || historyList.length === 0) {
            watchGrid.innerHTML = '<div class="empty-state">No movies watched yet.</div>';
            return;
        }
        
        watchGrid.innerHTML = "";
        historyList.forEach(item => {
            const card = document.createElement("div");
            card.className = "watch-card";
            
            const ratingBadge = item.rating 
                ? `<span class="user-rating-badge"><i class="fa-solid fa-star"></i> ${item.rating}/5</span>` 
                : '<span class="user-rating-badge text-muted">Unrated</span>';
                
            card.innerHTML = `
                <div class="watch-card-header">
                    <span class="watch-title" title="${item.title}">${item.title}</span>
                    <span class="watch-content-type">${item.content_type}</span>
                </div>
                <div class="watch-genre">${item.genre} | ${item.language}</div>
                <div class="progress-container">
                    <div class="progress-info">
                        <span>Progress</span>
                        <span>${item.progress_percentage}%</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" style="width: ${item.progress_percentage}%"></div>
                    </div>
                </div>
                <div class="watch-meta-footer">
                    <span>Watched: ${item.watch_date}</span>
                    ${ratingBadge}
                </div>
            `;
            watchGrid.appendChild(card);
        });
    }

    /**
     * Render search history table rows.
     */
    function renderSearchHistory(searchList) {
        if (!searchList || searchList.length === 0) {
            searchTableBody.innerHTML = '<tr><td colspan="4" class="text-center">No search logs found.</td></tr>';
            return;
        }
        
        searchTableBody.innerHTML = "";
        searchList.forEach(item => {
            const row = document.createElement("tr");
            
            const clickBadge = item.clicked_position !== -1 
                ? `<span class="badge badge-click">Result #${item.clicked_position}</span>`
                : '<span class="badge badge-neutral">No Click</span>';
                
            row.innerHTML = `
                <td><strong>"${item.query}"</strong></td>
                <td>${item.date}</td>
                <td>${item.results_returned} matches</td>
                <td>${clickBadge}</td>
            `;
            searchTableBody.appendChild(row);
        });
    }

    /**
     * Render ratings and review cards.
     */
    function renderRatingsHistory(ratingsList) {
        if (!ratingsList || ratingsList.length === 0) {
            ratingsGrid.innerHTML = '<div class="empty-state">No movie reviews or explicit ratings available.</div>';
            return;
        }
        
        ratingsGrid.innerHTML = "";
        ratingsList.forEach(item => {
            const card = document.createElement("div");
            card.className = "rating-card";
            
            let stars = "";
            for (let i = 1; i <= 5; i++) {
                stars += i <= item.rating ? '<i class="fa-solid fa-star"></i>' : '<i class="fa-regular fa-star"></i>';
            }
            
            card.innerHTML = `
                <div class="rating-card-header">
                    <strong title="${item.title}">${item.title}</strong>
                    <span class="sentiment-badge sentiment-${item.sentiment}">${item.sentiment}</span>
                </div>
                <div class="rating-stars">${stars}</div>
                <p class="rating-genre">${item.genre}</p>
                <div class="rating-review-text">"${item.review_text || 'No comments left'}"</div>
                <div class="watch-meta-footer">
                    <span>Reviewed: ${item.date}</span>
                </div>
            `;
            ratingsGrid.appendChild(card);
        });
    }

    /**
     * Render recommended movie card items.
     */
    function renderRecommendations(recsList) {
        if (!recsList || recsList.length === 0) {
            recsGrid.innerHTML = '<div class="empty-state">No recommendations generated.</div>';
            return;
        }
        
        recsGrid.innerHTML = "";
        recsList.forEach(movie => {
            const card = document.createElement("div");
            card.className = "movie-card";
            
            const netflixOriginal = movie.is_netflix_original 
                ? '<span class="netflix-badge">ORIGINAL</span>' 
                : "";
                
            const coverLetter = movie.title ? movie.title.charAt(0) : "N";
            
            card.innerHTML = `
                <div class="movie-cover">
                    ${netflixOriginal}
                    <span class="match-badge">${movie.match_score}% Match</span>
                    <span class="cover-letter">${coverLetter}</span>
                </div>
                <div class="movie-card-info">
                    <div>
                        <div class="movie-card-title" title="${movie.title}">${movie.title}</div>
                        <div class="movie-card-genres">${movie.genre_primary} | ${movie.genre_secondary}</div>
                    </div>
                    <div>
                        <div class="movie-card-metadata">
                            <span class="movie-rating-badge">${movie.rating}</span>
                            <span>${movie.release_year}</span>
                            <span>${movie.duration_minutes} min</span>
                            <span class="imdb-stars"><i class="fa-solid fa-star"></i> ${movie.imdb_rating}</span>
                        </div>
                        <button class="movie-card-action-btn" data-id="${movie.movie_id}">
                            <i class="fa-solid fa-circle-info"></i> More Like This
                        </button>
                    </div>
                </div>
            `;
            
            // Add click listener on details button
            const btn = card.querySelector(".movie-card-action-btn");
            btn.addEventListener("click", () => {
                showMovieDetails(movie.movie_id);
            });
            
            recsGrid.appendChild(card);
        });
    }

    /**
     * Fetch movie details and open Modal.
     */
    async function showMovieDetails(movieId) {
        modalBody.innerHTML = '<div class="loader">Loading movie metadata and similar content...</div>';
        modal.classList.remove("hidden");
        
        try {
            const response = await fetch(`/api/movie/${movieId}`);
            if (!response.ok) throw new Error("Failed to load movie details");
            const movie = await response.json();
            
            const warningAlert = movie.content_warning 
                ? '<div class="warning-banner"><i class="fa-solid fa-triangle-exclamation"></i> Content Warning: Contains mature themes.</div>' 
                : "";
                
            const originalBadge = movie.is_netflix_original 
                ? '<span class="user-id-badge">Netflix Original</span>' 
                : "";
                
            let similarCards = "";
            if (movie.similar_movies && movie.similar_movies.length > 0) {
                movie.similar_movies.forEach(sm => {
                    const smOrig = sm.is_netflix_original ? ' <span class="text-red" style="color: var(--accent-red); font-size: 0.7rem; font-weight:800;">[O]</span>' : '';
                    similarCards += `
                        <div class="similar-card">
                            <div class="similar-title">${sm.title}${smOrig}</div>
                            <div class="similar-meta">
                                <span>${sm.genre_primary} | ${sm.release_year}</span>
                                <span class="imdb-stars"><i class="fa-solid fa-star"></i> ${sm.imdb_rating}</span>
                            </div>
                        </div>
                    `;
                });
            } else {
                similarCards = '<div class="empty-state">No similar content found.</div>';
            }
            
            modalBody.innerHTML = `
                <div class="modal-hero">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <h2>${movie.title}</h2>
                        ${originalBadge}
                    </div>
                    <div class="modal-meta-grid">
                        <span><strong>Maturity:</strong> ${movie.rating}</span>
                        <span><strong>Year:</strong> ${movie.release_year}</span>
                        <span><strong>Duration:</strong> ${movie.duration_minutes} mins</span>
                        <span><strong>IMDb:</strong> <i class="fa-solid fa-star text-orange" style="color:var(--warning-orange)"></i> ${movie.imdb_rating}/10</span>
                    </div>
                    ${warningAlert}
                </div>
                
                <div class="modal-details-grid">
                    <div class="detail-item">
                        <h4>Primary Genre</h4>
                        <p>${movie.genre_primary}</p>
                    </div>
                    <div class="detail-item">
                        <h4>Secondary Genre</h4>
                        <p>${movie.genre_secondary}</p>
                    </div>
                    <div class="detail-item">
                        <h4>Language</h4>
                        <p>${movie.language}</p>
                    </div>
                    <div class="detail-item">
                        <h4>Country of Origin</h4>
                        <p>${movie.country}</p>
                    </div>
                </div>
                
                <div class="modal-similar-section">
                    <h3>More Like This</h3>
                    <div class="similar-grid">
                        ${similarCards}
                    </div>
                </div>
            `;
            
        } catch (err) {
            console.error(err);
            modalBody.innerHTML = '<div class="error-msg">Failed to load movie details. Close and try again.</div>';
        }
    }

    /**
     * Setup Tab switching mechanism.
     */
    function setupTabSwitching() {
        tabButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                // Remove active classes
                tabButtons.forEach(b => b.classList.remove("active"));
                tabContents.forEach(c => c.classList.remove("active"));
                
                // Add active state to clicked button
                btn.classList.add("active");
                
                // Show matching tab content
                const targetTab = btn.getAttribute("data-tab");
                document.getElementById(targetTab).classList.add("active");
            });
        });
    }

    /**
     * Setup closing logic for detail modal.
     */
    function setupModalClosing() {
        closeModalBtn.addEventListener("click", () => {
            modal.classList.add("hidden");
        });
        
        // Close modal when clicking overlay background
        const overlay = modal.querySelector(".modal-overlay");
        overlay.addEventListener("click", () => {
            modal.classList.add("hidden");
        });
        
        // Escape key closes modal
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && !modal.classList.contains("hidden")) {
                modal.classList.add("hidden");
            }
        });
    }
});
