-- Burger Reviews: Rating Prediction + Topic Insights --

This project analyzes burger restaurant reviews and predicts star ratings from review text. It combines a predictive text model (TF-IDF + Ridge Regression) with interpretable keyword/topic-based analysis to understand what factors are most associated with higher or lower ratings.

-- Project Overview --

The workflow filters businesses to only those in the Burgers category, cleans review text, removes rating “leak” phrases (example: “5 stars”), trains a TF-IDF + Ridge Regression model to predict star ratings from text, extracts influential attribute-related terms, and builds interpretable topic features using keyword counts (service, wait time, burger quality, etc.). The project also compares topic mentions between high-star and low-star reviews, runs regression models on topic features (including positive vs negative splits), and ranks burger businesses by predicted average rating.

-- Models Used --

Model 1: TF-IDF + Ridge Regression (Main Predictive Model)
Input: cleaned review text
Features: TF-IDF unigrams and bigrams
Model: Ridge regression
Output: predicted star ratings per review and per business

Model 2: Keyword Topic Models (Interpretability)
These models use keyword-count features for categories such as Service, Speed/Wait, Burger Quality, Fries/Sides, Price/Value, Cleanliness, Atmosphere, and Delivery/Takeout. Topic features are used to compare high vs low star reviews, estimate the net effect of each category, and split certain categories into positive vs negative keywords to measure their impacts separately.

-- Outputs --

The script saves the following plots:

topic_high_minus_low.png
Topic keyword mentions in high-star reviews minus low-star reviews

topic_effects_net.png
Regression coefficients showing the net effect of topic keyword mentions

topic_effects_pos_neg.png
Regression coefficients for positive vs negative topic keyword mentions

-- Data --

This project uses two CSV files: Data/businesses.csv and Data/reviews.csv.

⚠️ Note: The dataset is not included in this repository because the files were too large to upload to GitHub. To run the project locally, place the CSV files in a folder named Data/ in the project root:

Data/
businesses.csv
reviews.csv

-- Notes / Method Details --

Review text is cleaned to remove explicit rating phrases such as “5 stars” to prevent leakage. TF-IDF is used to convert text into numeric features that highlight terms that are frequent in a review but rarer across the full dataset. Keyword topic features are simple counts of predefined words/phrases and are mainly used for interpretability.

TF-IDF is used to convert text into numeric features that highlight terms that are frequent in a review but rarer across the full dataset.

Keyword topic features are simple counts of predefined words/phrases and are mainly used for interpretability.
