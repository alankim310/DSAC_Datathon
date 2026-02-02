"""
filter_categories.py — analyze individual category tags by splitting category strings into single tags.

Inputs:
- dataset/review_level_joined.csv (business_id, stars_review, categories)

Outputs:
- dataset/top_100_split_categories.csv (top 100 tags by review volume, sorted by std dev)

What it does:
- Filters to restaurants, computes business-level averages and review counts.
- Explodes comma-separated categories into individual tags and aggregates stats per tag.
"""



import pandas as pd
import numpy as np
from datetime import datetime
import re

print("="*80)
print("INDIVIDUAL CATEGORY ANALYSIS (SPLIT TAGS)")
print("="*80)
print(f"Start time: {datetime.now()}")

# ============================================================================
# 1. Load the joined dataset
# ============================================================================
print("[1/5] Loading review_level_joined.csv...")
# Using chunking to be safe with memory, specifically getting columns we need
chunk_size = 100000
chunks = []
try:
    for chunk in pd.read_csv('dataset/review_level_joined.csv', chunksize=chunk_size):
        chunks.append(chunk[['business_id', 'stars_review', 'categories']])
    df = pd.concat(chunks, ignore_index=True)
    print(f"   ✓ Loaded {len(df):,} reviews")
except FileNotFoundError:
    print("   Error: dataset/review_level_joined.csv not found.")
    exit(1)

# ============================================================================
# 2. Filter for restaurants
# ============================================================================
print("\n[2/5] Filtering for restaurants...")
df = df.dropna(subset=['categories'])
df = df[df['categories'].str.contains('Restaurants', case=False, na=False)]
print(f"   Filtered to {len(df):,} restaurant reviews")

# ============================================================================
# 3. Calculate Business-Level Statistics
# ============================================================================
print("\n[3/5] Calculating business-level stats...")
# We need the average rating and review count for EACH business first
business_stats = df.groupby(['business_id', 'categories']).agg(
    business_avg_rating=('stars_review', 'mean'),
    business_review_count=('stars_review', 'count')
).reset_index()

print(f"   Found {len(business_stats):,} unique restaurant businesses")

# ============================================================================
# 4. Explode Categories
# ============================================================================
print("\n[4/5] Exploding categories...")
# Split "Pizza, Italian, Restaurants" -> ["Pizza", "Italian", "Restaurants"]
# Using regex to handle potential spacing variations
business_stats['category_list'] = business_stats['categories'].astype(str).apply(lambda x: [s.strip() for s in x.split(',')])

# Explode the list into rows
exploded_df = business_stats.explode('category_list')
exploded_df = exploded_df.rename(columns={'category_list': 'single_category'})

# Remove empty strings if any
exploded_df = exploded_df[exploded_df['single_category'].str.len() > 0]

print(f"   Created {len(exploded_df):,} business-category pairs")

# ============================================================================
# 5. Aggregating by Single Category
# ============================================================================
print("\n[5/5] Aggregating stats by individual category...")

# Define a function for weighted average
def weighted_avg(x):
    total_weight = x['business_review_count'].sum()
    if total_weight == 0:
        return x['business_avg_rating'].mean()
    return np.average(x['business_avg_rating'], weights=x['business_review_count'])

# Group by the single category
category_stats = exploded_df.groupby('single_category').agg(
    num_businesses=('business_id', 'count'),
    num_reviews=('business_review_count', 'sum'),
    std_dev_business_ratings=('business_avg_rating', 'std'),
    average_rating=('business_avg_rating', 'mean') # Placeholder, we will overwrite with weighted
)

# Calculate weighted average (can be slower, so we do it carefully)
# Optimizing: sum(rating * count) / sum(count)
# We can do this vectorised easily
exploded_df['weighted_sum'] = exploded_df['business_avg_rating'] * exploded_df['business_review_count']
weighted_sums = exploded_df.groupby('single_category')['weighted_sum'].sum()
total_counts = exploded_df.groupby('single_category')['business_review_count'].sum()
category_stats['average_rating'] = weighted_sums / total_counts

# Reset index to make single_category a column
category_stats = category_stats.reset_index()

# Filter out "Restaurants" category if desired?
# The user said "category of restaurants", usually "Restaurants" is a generic tag. 
# It will obscure specific cuisines if left in top 1. 
# Identifying "Restaurants" tag specifically.
# Only removing strictly "Restaurants" tag, keeping "Pop-Up Restaurants" etc.
category_stats = category_stats[category_stats['single_category'] != 'Restaurants']

# Filter for Top 100 by volume
print("   Selecting Top 100 categories by review volume...")
top_100 = category_stats.sort_values('num_reviews', ascending=False).head(100).copy()

# Sort by Standard Deviation (Descending)
final_df = top_100.sort_values('std_dev_business_ratings', ascending=False)

# Save
output_file = 'dataset/top_100_split_categories.csv'
final_df.to_csv(output_file, index=False)
print(f"   ✓ Saved to {output_file}")

# Display
print("\n" + "="*80)
print("TOP 100 SINGLE CATEGORIES (Sorted by Std Dev)")
print("="*80)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)
pd.set_option('display.float_format', '{:.4f}'.format)

print(final_df[['single_category', 'num_reviews', 'num_businesses', 'average_rating', 'std_dev_business_ratings']].to_string(index=False))
print("\n" + "="*80)
print(f"End time: {datetime.now()}")
