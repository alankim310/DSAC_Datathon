
"""
Datathon-Winning-Code — Burger review modeling pipeline

Summary
- Loads business and review CSVs, filters to burger restaurants, and cleans review text (removes explicit star mentions).
- Trains a TF‑IDF + Ridge regression model to predict review star ratings from text.
- Reports attribute keyword terms that most increase/decrease predicted stars.
- Builds topic features (Service, Speed/Wait, Burger Quality, Fries/Sides, Price/Value, Cleanliness, Atmosphere, Delivery/Takeout) and compares mention rates in high vs low star reviews.
- Fits linear regressions using (a) net topic counts and (b) positive vs negative term counts to estimate impact on stars.
- Ranks businesses by average predicted stars from their review texts.
- Saves plots: topic_high_minus_low.png, topic_effects_net.png, topic_effects_pos_neg.png.

Inputs (expected columns)
- Data/businesses.csv: includes at least `business_id`, `categories`.
- Data/reviews.csv: includes at least `business_id`, `text`, `stars`.

Outputs
- Console summaries (model MAE/R^2, keyword/topic effects, top/bottom businesses).
- PNG plots saved to the working directory.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

#create keywords 

ATTRIBUTE_KEYWORDS = {
    "Service": [
        "service", "staff", "employee", "employees", "server", "waiter", "waitress",
        "manager", "host", "friendly", "rude", "attentive", "helpful"
    ],
    "Speed/Wait": [
        "wait", "waiting", "slow", "fast", "quick", "line", "minutes", "took forever",
        "long time", "busy"
    ],
    "Burger Quality": [
        "burger", "patty", "juicy", "dry", "seasoned", "bland", "greasy", "overcooked",
        "undercooked", "raw", "burnt", "fresh", "flavor", "taste"
    ],
    "Fries/Sides": [
        "fries", "french fries", "onion rings", "sides", "crispy", "soggy",
        "cold fries", "hot fries"
    ],
    "Price/Value": [
        "price", "expensive", "cheap", "overpriced", "worth", "value", "money", "portion"
    ],
    "Cleanliness": [
        "clean", "dirty", "gross", "sticky", "bathroom", "restroom"
    ],
    "Atmosphere": [
        "atmosphere", "vibe", "music", "loud", "quiet", "crowded", "seating", "ambience"
    ],
    "Delivery/Takeout": [
        "delivery", "doordash", "uber eats", "ubereats", "takeout", "pickup", "to go", "carryout"
    ],
}

TOPIC_SENTIMENT = {
    "Burger Quality": {
        "pos": ["juicy", "fresh", "tasty", "delicious", "perfect", "flavorful", "seasoned"],
        "neg": ["dry", "bland", "greasy", "overcooked", "undercooked", "raw", "burnt", "cold"]
    },
    "Service": {
        "pos": ["friendly", "helpful", "attentive", "nice", "kind"],
        "neg": ["rude", "ignored", "unfriendly", "disrespectful", "inattentive"]
    },
    "Speed/Wait": {
        "pos": ["fast", "quick"],
        "neg": ["wait", "waiting", "slow", "line", "took forever", "long time"]
    },
    "Fries/Sides": {
        "pos": ["crispy", "hot fries", "fresh"],
        "neg": ["cold fries", "soggy", "stale"]
    },
    "Price/Value": {
        "pos": ["worth", "value", "reasonable"],
        "neg": ["overpriced", "expensive", "not worth"]
    },
}

ATTR_TERMS = {w.lower() for words in ATTRIBUTE_KEYWORDS.values() for w in words}

#load data

business_data = pd.read_csv(r"Data\businesses.csv")
review_data = pd.read_csv(r"Data\reviews.csv")

business_data["categories"] = business_data["categories"].fillna("")
review_data["stars"] = pd.to_numeric(review_data["stars"], errors="coerce")
review_data = review_data.dropna(subset=["business_id", "text", "stars"]).copy()

burger_ids = set(business_data[business_data["categories"].str.contains
                               ("Burgers", case=False)]["business_id"])

burger_reviews = review_data[review_data["business_id"].isin(burger_ids)].copy()

print("Burger businesses:", len(burger_ids))
print("Burger reviews:", len(burger_reviews))

#clean text and get rid of leaks 

STAR_PHRASE_RE = re.compile(r"\b(one|two|three|four|five|\d)\s+stars?\b", flags=re.IGNORECASE)

def clean_text(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = STAR_PHRASE_RE.sub("", s)
    return s

burger_reviews["text_clean"] = burger_reviews["text"].apply(clean_text)


#TF-IDF + Ridge regression model

X = burger_reviews["text_clean"]
y = burger_reviews["stars"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        min_df=3
    )),
    ("model", Ridge(alpha=2.0))
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

print("\n--- MODEL PERFORMANCE (burger reviews) ---")
print("MAE:", mean_absolute_error(y_test, pred))
print("R^2:", r2_score(y_test, pred))


#Filter out only attributed-related words  

feature_names = pipe.named_steps["tfidf"].get_feature_names_out()
coefs = pipe.named_steps["model"].coef_
coef_df = pd.DataFrame({"term": feature_names, "coef": coefs})

coef_attr_only = coef_df[coef_df["term"].isin(ATTR_TERMS)].copy()

print("\nAttribute-related terms that INCREASE predicted stars:")
print(coef_attr_only.sort_values("coef", ascending=False).head(30).to_string(index=False))

print("\nAttribute-related terms that DECREASE predicted stars:")
print(coef_attr_only.sort_values("coef", ascending=True).head(30).to_string(index=False))



#helper function for counting keywords
def count_terms(text: str, terms) -> int:
    text = str(text).lower()
    total = 0
    for t in terms:
        total += len(re.findall(r"\b" + re.escape(t.lower()) + r"\b", text))
    return total


#high vs low stars topic comparison

topic_cols = list(ATTRIBUTE_KEYWORDS.keys())

topic_df = pd.DataFrame({
    topic: burger_reviews["text_clean"].apply(lambda t, ws=words: count_terms(t, ws))
    for topic, words in ATTRIBUTE_KEYWORDS.items()
})

topic_model_df = pd.concat([burger_reviews[["stars"]].reset_index(drop=True),
                            topic_df.reset_index(drop=True)], axis=1)

# only reviews mentioning at least 1 topic keyword

topic_model_df["topic_total_mentions"] = topic_model_df[topic_cols].sum(axis=1)
topic_model_df = topic_model_df[topic_model_df["topic_total_mentions"] > 0].copy()

high = topic_model_df[topic_model_df["stars"] >= 4.5]
low  = topic_model_df[topic_model_df["stars"] <= 2.0]

topic_summary = pd.DataFrame({
    "high_avg_mentions": high[topic_cols].mean(),
    "low_avg_mentions": low[topic_cols].mean()
})
topic_summary["difference(high-low)"] = topic_summary["high_avg_mentions"] - topic_summary["low_avg_mentions"]
topic_summary = topic_summary.sort_values("difference(high-low)")

print("\n--- Topic mention comparison (High vs Low stars) ---")
print(topic_summary.to_string())

#plot net impact of topics 

plt.figure(figsize=(10, 6))
plt.barh(topic_summary.index, topic_summary["difference(high-low)"].values)
plt.title("Topic Mentions: High-Star minus Low-Star Reviews")
plt.xlabel("Avg mentions per review (High - Low)")
plt.tight_layout()
plt.savefig("topic_high_minus_low.png", dpi=300, bbox_inches="tight")
plt.show()



# creating keyword-based topic regression model to easier
# visualize the data and create readable plots for presentation

X_topics = topic_model_df[topic_cols]
y_stars = topic_model_df["stars"]

Xt_train, Xt_test, yt_train, yt_test = train_test_split(
    X_topics, y_stars, test_size=0.2, random_state=42
)

topic_reg = LinearRegression()
topic_reg.fit(Xt_train, yt_train)
topic_pred = topic_reg.predict(Xt_test)

topic_effects = pd.Series(topic_reg.coef_, index=topic_cols).sort_values()

print("\nTopic regression R^2:", r2_score(yt_test, topic_pred))
print("\nTopic effects (NET coef):")
print(topic_effects.to_string())

plt.figure(figsize=(10, 6))
plt.barh(topic_effects.index, topic_effects.values)
plt.title("Category-Level Impact on Stars (Topic Net Coefficients)")
plt.xlabel("Effect on Stars per 1 Keyword Mention (net; can cancel)")
plt.tight_layout()
plt.savefig("topic_effects_net.png", dpi=300, bbox_inches="tight")
plt.show()


#Plotting pos vs neg impact instead of net impact 

pn_features = {}
for topic, d in TOPIC_SENTIMENT.items():
    pn_features[f"{topic}_pos"] = burger_reviews["text_clean"].apply
    (lambda t, ws=d["pos"]: count_terms(t, ws))
    pn_features[f"{topic}_neg"] = burger_reviews["text_clean"].apply
    (lambda t, ws=d["neg"]: count_terms(t, ws))

topic_pn_df = pd.DataFrame(pn_features)
topic_pn_df["stars"] = burger_reviews["stars"].values

pn_cols = [c for c in topic_pn_df.columns if c != "stars"]
topic_pn_df["total_mentions"] = topic_pn_df[pn_cols].sum(axis=1)
topic_pn_df = topic_pn_df[topic_pn_df["total_mentions"] > 0].copy()

Xp = topic_pn_df[pn_cols]
yp = topic_pn_df["stars"]

Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    Xp, yp, test_size=0.2, random_state=42
)

pn_reg = LinearRegression()
pn_reg.fit(Xp_train, yp_train)
pn_pred = pn_reg.predict(Xp_test)

pn_effects = pd.Series(pn_reg.coef_, index=pn_cols).sort_values()

print("\nPos/Neg topic regression R^2:", r2_score(yp_test, pn_pred))
print("\nTopic effects (POS vs NEG coef):")
print(pn_effects.to_string())

plt.figure(figsize=(10, 8))
plt.barh(pn_effects.index, pn_effects.values)
plt.title("Positive vs Negative Topic Impact on Stars (Coefficients)")
plt.xlabel("Effect on Stars per 1 Keyword Mention")
plt.tight_layout()
plt.savefig("topic_effects_pos_neg.png", dpi=300, bbox_inches="tight")
plt.show()


#Get top and bottom businesses by predicted stars from review text 

burger_reviews["predicted_review_stars"] = pipe.predict(burger_reviews["text_clean"])

business_pred = (burger_reviews.groupby("business_id")
                 .agg(predicted_business_stars=("predicted_review_stars", "mean"),
                      true_avg_review_stars=("stars", "mean"),
                      n_reviews=("stars", "size"))
                 .reset_index())

print("\nTop 15 businesses by predicted avg stars (from review text):")
print(business_pred.sort_values("predicted_business_stars", ascending=False).head(15).to_string(index=False))

print("\nBottom 15 businesses by predicted avg stars (from review text):")
print(business_pred.sort_values("predicted_business_stars", ascending=True).head(15).to_string(index=False))


print("\nSaved plots:")
print(" - topic_high_minus_low.png")
print(" - topic_effects_net.png")
print(" - topic_effects_pos_neg.png")

