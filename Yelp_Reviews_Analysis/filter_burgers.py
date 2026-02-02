"""
filter_burgers.py — create a burger-only joined dataset by filtering businesses and reviews.

Inputs:
- dataset/businesses.csv
- dataset/reviews.csv

Outputs:
- modified/burgers_joined.csv

What it does:
- Filters businesses to those with the "Burgers" category.
- Filters reviews to those businesses and joins with business info.
"""



import pandas as pd
from datetime import datetime

print("="*80)
print("JOIN BUSINESSES AND REVIEWS FOR BURGERS CATEGORY")
print("="*80)
print(f"Start time: {datetime.now()}")

# ============================================================================
# 1. Load and filter businesses for Burgers
# ============================================================================
print("\n[1/3] Loading businesses.csv and filtering for 'Burgers'...")
businesses = pd.read_csv('dataset/businesses.csv')
print(f"   Total businesses: {len(businesses):,}")

# Filter for businesses with 'Burgers' in categories
businesses_burgers = businesses[
    businesses['categories'].notna() & 
    businesses['categories'].str.contains('Burgers', case=False, na=False)
].copy()

print(f"   Businesses with 'Burgers' category: {len(businesses_burgers):,}")

# Get the business_ids for filtering
burger_business_ids = set(businesses_burgers['business_id'])
print(f"   Unique burger business IDs: {len(burger_business_ids):,}")

# ============================================================================
# 2. Load reviews in chunks and filter
# ============================================================================
print("\n[2/3] Loading reviews.csv in chunks and filtering...")
chunk_size = 100000
filtered_reviews = []
total_reviews_processed = 0

for i, chunk in enumerate(pd.read_csv('dataset/reviews.csv', chunksize=chunk_size)):
    total_reviews_processed += len(chunk)
    # Filter for reviews that match burger business_ids
    filtered_chunk = chunk[chunk['business_id'].isin(burger_business_ids)]
    if len(filtered_chunk) > 0:
        filtered_reviews.append(filtered_chunk)
    
    if (i + 1) % 10 == 0:
        print(f"   Processed {total_reviews_processed:,} reviews...")

reviews_burgers = pd.concat(filtered_reviews, ignore_index=True) if filtered_reviews else pd.DataFrame()

print(f"   Total reviews processed: {total_reviews_processed:,}")
print(f"   Burger-related reviews: {len(reviews_burgers):,}")

# ============================================================================
# 3. Join the datasets
# ============================================================================
print("\n[3/3] Joining businesses and reviews...")
joined_df = reviews_burgers.merge(
    businesses_burgers,
    on='business_id',
    how='inner'
)

print(f"   Joined dataset size: {len(joined_df):,} rows")
print(f"   Columns: {len(joined_df.columns)}")

# Save the result
output_file = 'modified/burgers_joined.csv'
joined_df.to_csv(output_file, index=False)
print(f"   ✓ Saved to {output_file}")

# Display summary statistics
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Unique businesses: {joined_df['business_id'].nunique():,}")
print(f"Total reviews: {len(joined_df):,}")
print(f"Average rating: {joined_df['stars_x'].mean():.2f}" if 'stars_x' in joined_df.columns else "Average rating: N/A")
print(f"Date range: {joined_df['date'].min()} to {joined_df['date'].max()}" if 'date' in joined_df.columns else "")

print("\n" + "="*80)
print(f"End time: {datetime.now()}")
