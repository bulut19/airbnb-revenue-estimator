# Austin Airbnb: Revenue Potential Estimator

A gradient boosting model that predicts how much annual revenue an Austin Airbnb listing can expect to earn, given its specs, location, host profile, and nightly price.

## Live Demo

**[austin-airbnb-revenue-estimator.melisa-data.workers.dev](https://austin-airbnb-revenue-estimator.melisa-data.workers.dev/#try)**

Enter a property's specs and a nightly price to get an estimated occupancy and annual revenue.

## Overview

Airbnb calculates revenue as price × occupancy rather than measuring it directly, so this model follows the same logic: predict nights booked per year with a gamma-loss `HistGradientBoostingRegressor`, then multiply by the nightly price the host supplies. Built on the Inside Airbnb Austin TX dataset (June 2026 snapshot, 11,295 raw listings filtered to 6,537 with sufficient review history and valid pricing).

137 engineered features across property specs, crowding ratios, location, host profile, and 114 individual amenity flags. Review counts, review scores, and availability were deliberately excluded since Airbnb derives its occupancy estimate from those same signals, including them would leak the answer into the input.

| | R² (log scale) | R² (dollar scale) | MAE | Median % error |
|---|---|---|---|---|
| **Final model** | **0.547** | **0.552** | **$14,843** | **39.3%** |

## Key Findings

- Gamma loss beat every alternative tested: squared-error, Poisson, linear regression, and a tuned random forest, by treating percentage error consistently across revenue levels rather than assuming a fixed dollar error.
- Accuracy varies by listing size: median error is ~130% for the lowest-earning quintile of listings and ~38% for the highest-earning quintile.
- Predictions are conditional on the price entered. The model estimates occupancy at that price, it does not recommend a price.

## Tech Stack

Python (pandas, NumPy, scikit-learn) · Google Colab · Cloudflare Workers (live demo deployment) · Claude (Sonnet 5)

## Repository Contents

- `airbnb_revenue_estimator.ipynb`: full notebook (preprocessing, feature engineering, model selection, evaluation, and an interactive prediction demo)
- `README.md`: this file
