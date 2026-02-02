# Datathon Project

This repository contains two main components:
- Machine learning analysis of burger reviews.
- Yelp review data wrangling and exploratory analysis for Philadelphia restaurants.

Sub-readmes:
- Machine learning details: [Machine_Learning/README.md](Machine_Learning/README.md)
- Yelp review analysis details: [Yelp_Reviews_Analysis/README.md](Yelp_Reviews_Analysis/README.md)

## Machine Learning (Burger Reviews)
The Machine_Learning directory builds a text-based rating predictor for burger reviews (TF-IDF + Ridge Regression) and interprets drivers of ratings using keyword/topic features (service, speed, burger quality, fries, price/value, cleanliness, atmosphere, delivery/takeout). It also compares high vs low star topic mentions and ranks businesses by predicted stars. See [Machine_Learning/README.md](Machine_Learning/README.md) for datasets, outputs, and plots.

## Yelp Reviews Analysis (Data Wrangling)
The Yelp_Reviews_Analysis directory builds a review-level joined dataset, analyzes restaurant category variance (full category strings and split tags), creates burger-only joins, and runs a Philadelphia neighborhood value analysis. See [Yelp_Reviews_Analysis/README.md](Yelp_Reviews_Analysis/README.md) for scripts and outputs.

## Datathon Problem Statement (Summary)
The challenge asked teams to analyze Yelp data for Philadelphia restaurants and identify factors most strongly associated with higher Yelp ratings, producing actionable recommendations a restaurant owner could realistically implement. The expected deliverable was a 5–10 minute presentation explaining the question/hypothesis, data usage, key findings, and business recommendations.

## Presentation Summary
Our presentation focused on understanding what drives burger restaurant ratings in Philadelphia. We showed that quality varies substantially, built a review-text model to quantify keyword impacts, and found that negative experiences have a stronger effect on ratings than positive ones. Service quality (avoiding rude or inattentive staff) and food quality were highlighted as major drivers of higher ratings. We discussed model improvements (moving from a weak baseline to substantially better performance) and acknowledged limitations such as correlation vs causation and keyword bias. The recommendations emphasized cleaning up negative drivers, improving service standards, and strengthening burger craftsmanship, along with location considerations for expansion.
