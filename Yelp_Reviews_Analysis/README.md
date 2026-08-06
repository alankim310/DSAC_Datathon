# Yelp Review Data Pipeline & Market Analysis

This component prepares large Yelp CSV files for analysis and creates restaurant-level views of category performance, burger reviews, and Philadelphia neighborhood value.

## What this work enables

Raw Yelp exports are split across businesses, users, check-ins, tips, and reviews. These scripts turn them into analysis-ready tables, then answer questions such as:

- Which restaurant categories have the widest variation in business ratings?
- How do individual cuisine tags compare once multi-category businesses are separated?
- What does the burger-restaurant market look like in Philadelphia?
- Which neighborhoods appear promising when customer ratings are considered alongside estimated rent?

## Pipeline at a glance

```text
Yelp source CSVs
      |
      v
join_datasets.py  --> review-level joined dataset
      |                         |
      |                         +--> category analyses
      v
filter_burgers.py --> burger-only review dataset --> neighborhood value analysis
```

## Scripts and outputs

| Script | Purpose | Primary output |
| --- | --- | --- |
| [`join_datasets.py`](join_datasets.py) | Joins businesses, users, check-ins, tips, and reviews; reads large files in chunks | `dataset/review_level_joined.csv` and a 1,000-row sample |
| [`filter_restaurant_reviews.py`](filter_restaurant_reviews.py) | Summarizes full restaurant category strings by review volume, average rating, and rating variation | `dataset/restaurant_category_analysis.csv` |
| [`filter_categories.py`](filter_categories.py) | Splits multi-category strings into individual tags and summarizes the top 100 by review volume | `dataset/top_100_split_categories.csv` |
| [`filter_burgers.py`](filter_burgers.py) | Filters businesses tagged with Burgers and joins matching reviews | `modified/burgers_joined.csv` |
| [`philly_neighborhood_analysis.py`](philly_neighborhood_analysis.py) | Calculates a Philadelphia burger-restaurant value score using ratings and estimated rent | `philly-neighborhood-analysis.csv` |

## Data requirements

The source data and large generated files are intentionally excluded from GitHub. Place the Yelp exports here before running the pipeline:

```text
Yelp_Reviews_Analysis/
└── dataset/
    ├── businesses.csv
    ├── users.csv
    ├── checkins.csv
    ├── tips.csv
    └── reviews.csv
```

`philly_neighborhood_analysis.py` expects a burger dataset named `burgers.csv`; update its input path if you want to use the output from `filter_burgers.py` (`modified/burgers_joined.csv`).

## Suggested run order

```bash
cd Yelp_Reviews_Analysis
python join_datasets.py
python filter_restaurant_reviews.py
python filter_categories.py
python filter_burgers.py
python philly_neighborhood_analysis.py
```

## Implementation notes

- Large review files are processed in chunks where appropriate to make the pipeline more practical on a local machine.
- Category analysis reports both volume and variation in business ratings, helping distinguish well-sampled patterns from small-sample noise.
- This project analyzes historical Yelp data. Results should be treated as exploratory signals, not as causal estimates or current market conditions.

For the modeling and business interpretation built on the burger subset, see the [machine learning analysis](../Machine_Learning/README.md).
