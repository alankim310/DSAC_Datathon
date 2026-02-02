# Yelp Reviews Analysis

This directory contains data-wrangling and analysis scripts for Yelp restaurant data, with a focus on category variance, burger-specific joins, and a Philadelphia neighborhood value analysis. Note that the repository does not contain CSV data as the data is huge. 

## What this directory does
- Builds a large, review-level joined dataset across businesses, users, checkins, tips, and reviews.
- Analyzes restaurant category variability (both full-category strings and split tags).
- Produces top-category reports by review volume and rating variance.
- Builds a burger-only dataset and runs a Philadelphia neighborhood “value score” analysis.

## Data layout
- dataset/ contains the original (large) Yelp CSVs and generated outputs. These files are not included in the repository because they are too large.
- All example datasets for this analysis are stored in the dataset/ subdirectory.
- modified/ contains derived CSVs from analyses (kept as snapshots).
- Root-level CSVs are ad-hoc or exported results.

## Script overview
- filter_restaurant_reviews.py
  - Input: dataset/review_level_joined.csv
  - Output: dataset/restaurant_category_analysis.csv
  - Purpose: computes per-category averages, counts, and business-rating standard deviation (full category strings).

- filter_categories.py
  - Input: dataset/review_level_joined.csv
  - Output: dataset/top_100_split_categories.csv
  - Purpose: splits category strings into individual tags and computes stats per tag.

- join_datasets.py
  - Inputs: dataset/businesses.csv, dataset/users.csv, dataset/checkins.csv, dataset/tips.csv, dataset/reviews.csv
  - Outputs: dataset/review_level_joined.csv, dataset/review_level_sample.csv
  - Purpose: creates a review-level joined dataset (large).
  - review_level_joined.csv is not in the repository, because the data size is too big (5GB). 

- filter_burgers.py
  - Inputs: dataset/businesses.csv, dataset/reviews.csv
  - Output: modified/burgers_joined.csv
  - Purpose: filters businesses and reviews for the Burgers category and joins them.

- philly_neighborhood_analysis.py
  - Input: burgers.csv
  - Output: philly-neighborhood-analysis.csv
  - Purpose: computes a neighborhood “value score” for Philly burger spots using ratings and estimated rent.

## Typical workflow
1. Run join_datasets.py to create the main joined table.
2. Run filter_restaurant_reviews.py and filter_top_categories.py for full-category analysis.
3. Run filter_categories.py for tag-level analysis.
4. Run filter_burgers.py to generate burger-only joins.
5. Run philly_neighborhood_analysis.py on burgers.csv (or adjust to use modified/burgers_joined.csv).

