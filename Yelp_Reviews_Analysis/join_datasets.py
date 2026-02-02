"""
Join Yelp datasets into a single review-level table.

Inputs:
- dataset/businesses.csv
- dataset/users.csv
- dataset/checkins.csv
- dataset/tips.csv
- dataset/reviews.csv

Outputs:
- dataset/review_level_joined.csv
- dataset/review_level_sample.csv

What it does:
- Loads large CSVs in chunks, joins reviews with businesses, users, checkins, and tips.
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("Starting data joining process...")
print(f"Current time: {datetime.now()}")

# Define chunk size for large files
chunk_size = 100000

# ============================================================================
# 1. Load businesses.csv (smaller file - load fully)
# ============================================================================
print("\n[1/5] Loading businesses.csv...")
businesses = pd.read_csv('dataset/businesses.csv')
print(f"   Loaded {len(businesses):,} businesses")

# ============================================================================
# 2. Load users.csv (load in chunks due to size)
# ============================================================================
print("\n[2/5] Loading users.csv in chunks...")
user_chunks = []
for i, chunk in enumerate(pd.read_csv('dataset/users.csv', chunksize=chunk_size)):
    user_chunks.append(chunk)
    if (i + 1) % 10 == 0:
        print(f"   Processed {(i + 1) * chunk_size:,} users...")
users = pd.concat(user_chunks, ignore_index=True)
print(f"   Loaded {len(users):,} users")

# ============================================================================
# 3. Process checkins.csv - aggregate by business_id
# ============================================================================
print("\n[3/5] Loading and aggregating checkins.csv...")
checkins = pd.read_csv('dataset/checkins.csv')
# Count number of checkin dates per business
checkins['checkin_count'] = checkins['date'].str.split(',').str.len()
checkins_agg = checkins[['business_id', 'checkin_count']]
print(f"   Loaded {len(checkins_agg):,} checkin records")

# ============================================================================
# 4. Load tips.csv (smaller file)
# ============================================================================
print("\n[4/5] Loading tips.csv...")
tips = pd.read_csv('dataset/tips.csv')
print(f"   Loaded {len(tips):,} tips")

# ============================================================================
# 5. Load reviews.csv in chunks and join incrementally
# ============================================================================
print("\n[5/5] Loading reviews.csv and joining with other datasets...")
print("   This is the largest file and will take some time...")

joined_chunks = []
review_count = 0

for i, review_chunk in enumerate(pd.read_csv('dataset/reviews.csv', chunksize=chunk_size)):
    review_count += len(review_chunk)
    
    # Join with businesses
    chunk_joined = review_chunk.merge(
        businesses,
        on='business_id',
        how='left',
        suffixes=('_review', '_business')
    )
    
    # Join with users
    chunk_joined = chunk_joined.merge(
        users,
        on='user_id',
        how='left',
        suffixes=('', '_user')
    )
    
    # Join with checkins (aggregated)
    chunk_joined = chunk_joined.merge(
        checkins_agg,
        on='business_id',
        how='left'
    )
    
    # Join with tips
    chunk_joined = chunk_joined.merge(
        tips[['business_id', 'user_id', 'text', 'date', 'compliment_count']],
        on=['business_id', 'user_id'],
        how='left',
        suffixes=('', '_tip')
    )
    
    joined_chunks.append(chunk_joined)
    
    if (i + 1) % 5 == 0:
        print(f"   Processed {review_count:,} reviews...")

# Combine all chunks
print("\n[6/6] Combining all chunks into final dataset...")
final_df = pd.concat(joined_chunks, ignore_index=True)

print("\n" + "="*70)
print("JOIN COMPLETE!")
print("="*70)
print(f"Total reviews (rows): {len(final_df):,}")
print(f"Total columns: {len(final_df.columns)}")
print(f"\nColumn names:")
for col in final_df.columns:
    print(f"  - {col}")

# ============================================================================
# Save the result
# ============================================================================
print("\n" + "="*70)
print("Saving results...")
print("="*70)

# Save to CSV (might be large)
output_file = 'dataset/review_level_joined.csv'
print(f"\nSaving to {output_file}...")
final_df.to_csv(output_file, index=False)
print(f"✓ Saved successfully!")

# Also save a sample for quick inspection
sample_file = 'dataset/review_level_sample.csv'
print(f"\nSaving sample (first 1000 rows) to {sample_file}...")
final_df.head(100).to_csv(sample_file, index=False)
print(f"✓ Sample saved!")

# ============================================================================
# Display summary statistics
# ============================================================================
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"\nDataset shape: {final_df.shape}")
print(f"Memory usage: {final_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\nMissing values by column:")
missing = final_df.isnull().sum()
missing_pct = (missing / len(final_df)) * 100
missing_df = pd.DataFrame({
    'Missing': missing,
    'Percentage': missing_pct
}).sort_values('Missing', ascending=False)
print(missing_df[missing_df['Missing'] > 0].head(10))

print("\n" + "="*70)
print("DONE!")
print("="*70)
print(f"\nFinal time: {datetime.now()}")

print(sample_file)