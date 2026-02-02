"""
filter_restaurant_reviews.py — compute restaurant category variability from the joined review-level dataset.

Inputs:
- dataset/review_level_joined.csv (uses business_id, stars_review, categories)

Outputs:
- dataset/restaurant_category_analysis.csv (per-category avg rating, review/business counts,
  std dev of business average ratings)

What it does:
- Filters to rows whose categories include "Restaurants".
- Computes business-level averages, then aggregates per category.
"""


import pandas as pd
import numpy as np
from datetime import datetime

print("="*80)
print("CUISINE CATEGORY ANALYSIS")
print("="*80)
print(f"Start time: {datetime.now()}")
print()

# ============================================================================
# 1. Load the joined dataset
# ============================================================================
print("[1/4] Loading review_level_joined.csv...")
print("   This may take a moment due to file size...")

chunk_size = 100000
chunks = []
for chunk in pd.read_csv('dataset/review_level_joined.csv', chunksize=chunk_size):
    # Keep only columns we need: business_id, stars (from review), categories
    chunk_subset = chunk[['business_id', 'stars_review', 'categories']].copy()
    chunks.append(chunk_subset)

df = pd.concat(chunks, ignore_index=True)
print(f"   ✓ Loaded {len(df):,} reviews")

# ============================================================================
# 2. Remove rows with missing categories and filter for restaurants only
# ============================================================================
print("\n[2/4] Cleaning data and filtering for restaurants...")
initial_count = len(df)
df = df.dropna(subset=['categories'])
print(f"   Removed {initial_count - len(df):,} reviews with missing categories")

# Filter for categories containing "Restaurants" (case-insensitive)
df = df[df['categories'].str.contains('Restaurants', case=False, na=False)]
print(f"   Filtered for restaurants only: {len(df):,} reviews remaining")

# ============================================================================
# 3. Calculate metrics for each unique category
# ============================================================================
print("\n[3/4] Calculating metrics for each unique category...")
print("   Step 1: Calculating average rating per business...")

# First, calculate average rating for each business
business_avg = df.groupby('business_id').agg({
    'stars_review': 'mean',
    'categories': 'first'  # Get the category (should be same for each business)
}).reset_index()
business_avg.columns = ['business_id', 'avg_business_rating', 'categories']

print(f"   ✓ Calculated averages for {len(business_avg):,} unique businesses")

print("   Step 2: Calculating standard deviation per category...")

# Now group by category and calculate:
# - Count of reviews (from original df)
# - Average rating across all reviews in category
# - Standard deviation of business average ratings
category_stats = []

for category in df['categories'].unique():
    # Get all reviews for this category
    category_reviews = df[df['categories'] == category]
    
    # Get all businesses for this category
    category_businesses = business_avg[business_avg['categories'] == category]
    
    # Calculate metrics
    num_reviews = len(category_reviews)
    avg_rating = category_reviews['stars_review'].mean()
    std_dev = category_businesses['avg_business_rating'].std()
    num_businesses = len(category_businesses)
    
    category_stats.append({
        'category': category,
        'average_rating': avg_rating,
        'num_reviews': num_reviews,
        'num_businesses': num_businesses,
        'std_dev_business_ratings': std_dev
    })

# Create DataFrame
results_df = pd.DataFrame(category_stats)

# Sort by standard deviation (descending)
results_df = results_df.sort_values('std_dev_business_ratings', ascending=False)

print(f"   ✓ Calculated metrics for {len(results_df):,} unique categories")

# ============================================================================
# 4. Save results and display
# ============================================================================
print("\n[4/4] Saving results...")
output_file = 'dataset/restaurant_category_analysis.csv'
results_df.to_csv(output_file, index=False)
print(f"   ✓ Saved to {output_file}")

# ============================================================================
# Display results
# ============================================================================
print("\n" + "="*80)
print("TOP 10 RESTAURANT CATEGORIES BY STANDARD DEVIATION OF BUSINESS RATINGS")
print("="*80)
print()

top_10 = results_df.head(10)
for idx, row in top_10.iterrows():
    print(f"Rank #{top_10.index.get_loc(idx) + 1}")
    print(f"  Category: {row['category']}")
    print(f"  Average Rating: {row['average_rating']:.2f}")
    print(f"  Number of Reviews: {row['num_reviews']:,}")
    print(f"  Number of Businesses: {row['num_businesses']:,}")
    print(f"  Std Dev (Business Ratings): {row['std_dev_business_ratings']:.4f}")
    print()

# ============================================================================
# Display full results
# ============================================================================
print("="*80)
print("ALL RESTAURANT CATEGORIES (sorted by standard deviation)")
print("="*80)
print()

# Set pandas display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 60)

print(results_df.to_string(index=False))

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Total unique restaurant categories: {len(results_df):,}")
print(f"Average std dev across all restaurant categories: {results_df['std_dev_business_ratings'].mean():.4f}")
print(f"Median std dev: {results_df['std_dev_business_ratings'].median():.4f}")
print(f"Max std dev: {results_df['std_dev_business_ratings'].max():.4f}")
print(f"Min std dev: {results_df['std_dev_business_ratings'].min():.4f}")

print("\n" + "="*80)
print("DONE!")
print("="*80)
print(f"End time: {datetime.now()}")
