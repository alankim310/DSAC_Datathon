# Burger Review Rating Prediction

This component uses Yelp review text to estimate star ratings for Philadelphia burger restaurants, then converts the model's signals into themes a restaurant owner can act on.

## Business question

Which parts of the burger-restaurant experience show up most consistently in high- and low-rated reviews?

## Approach

1. Filter Yelp businesses to those tagged with **Burgers** and join them to their reviews.
2. Clean review text and remove explicit rating phrases (for example, "5 stars") to avoid target leakage.
3. Train a **TF-IDF + Ridge regression** model to predict review star ratings from unigrams and bigrams.
4. Inspect influential terms and aggregate keyword mentions into understandable topics: service, speed/wait, burger quality, fries/sides, price/value, cleanliness, atmosphere, and delivery/takeout.
5. Compare topic frequency in high- versus low-star reviews and estimate positive and negative topic effects.
6. Aggregate predicted review ratings to rank businesses by predicted average rating.

## Why this approach is useful

The predictive model captures a wide range of review language, while the topic analysis explains the output in business terms. This makes it possible to move from "the model predicts a lower rating" to a concrete question such as whether reviews mention slow service, poor food quality, or weak value.

## Outputs

Running [`main.py`](main.py) produces model metrics in the console and saves these visualizations:

| Visualization | What it shows |
| --- | --- |
| [Topic mentions: high vs. low ratings](topic_high_minus_low.png) | Difference in topic-keyword mentions between high-star and low-star reviews |
| [Net topic effects](topic_effects_net.png) | Estimated relationship between each topic and rating |
| [Positive vs. negative topic effects](topic_effects_pos_neg.png) | Separate estimated effects for favorable and unfavorable topic language |

## Data requirements

The Yelp CSV files are not committed because of their size. To run the analysis, create this directory at the repository root:

```text
Data/
├── businesses.csv
└── reviews.csv
```

The script expects those exact paths. The necessary Python packages are `pandas`, `numpy`, `scikit-learn`, and `matplotlib`.

## Run

From the repository root:

```bash
python Machine_Learning/main.py
```

## Interpretation notes

- Model and topic coefficients describe associations in review data; they do not prove that a topic causes a rating change.
- Keyword topics are intentionally transparent rather than exhaustive, so they are best used as a guide for investigating operations and customer feedback.
- The full competition narrative and recommendations are available in the [data presentation](../assets/Data-Presentation.pdf).
