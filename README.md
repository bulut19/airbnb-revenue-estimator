# Austin Airbnb: Revenue Potential Estimator

A gradient boosting model that predicts how much annual revenue an Austin Airbnb listing can expect to earn, given its specs, location, host profile, and nightly price. Deployed as a live interactive tool.

## Live Demo

**[austin-airbnb-revenue-estimator.melisa-data.workers.dev](https://austin-airbnb-revenue-estimator.melisa-data.workers.dev/#try)**

Enter a property's specs and a nightly price to get an estimated occupancy and annual revenue, powered by the trained model below.

## Overview

**Question.** Given a property's specs, its location, the host's profile, and a nightly price, what annual revenue can the listing be expected to earn?

**Data.** Inside Airbnb, Austin TX, June 2026 snapshot (11,295 raw listings).

**Answer.** Predict nights booked per year with a gamma-loss gradient boosting model, then multiply by the nightly price the host supplies. Airbnb does not measure revenue directly, it calculates it the same way (price × occupancy), so the model is built around that same logic rather than predicting revenue as a single number.

| | R² (log scale) | R² (dollar scale) | MAE | Median % error |
|---|---|---|---|---|
| **Final model** | **0.547** | **0.552** | **$14,843** | **39.3%** |

## Approach

1. **Filtering.** Kept listings with 5+ reviews (Airbnb's own occupancy estimates need a minimum review history to be trustworthy), a valid price, and positive revenue. Left 6,537 listings from the original 11,295.
2. **Missing value handling.** `bathrooms` parsed from the free-text `bathrooms_text` field where the numeric column was missing; `bedrooms` and `beds` filled with the median for listings of the same size and room type.
3. **Feature engineering.** 137 features across six groups: property specs, crowding ratios (people per bedroom, beds per bedroom, bath per person), location (including haversine distance to downtown), host profile, categorical splits (`is_entire_place` + `dwelling` type), and 114 individual amenity flags, kept only where the amenity appears in 5-95% of listings.
4. **Target leakage avoidance.** Deliberately excluded review counts/scores and availability fields, since Airbnb's occupancy estimate is itself derived from review activity, using those features would let the model see a shortcut version of the answer.
5. **Model selection.** Tested squared-error loss, Poisson loss, linear regression, and a tuned random forest against a gamma-loss `HistGradientBoostingRegressor`. Gamma loss won by treating a given percentage error the same at every revenue level, which fits this data (errors scale with listing size) better than a fixed-dollar-error assumption.
6. **Evaluation.** 5-fold cross-validation repeated twice, scored on both log and dollar scales. Grouped permutation importance (shuffling correlated features like `accommodates`/`beds`/`bedrooms` together as a block) to get meaningful importance scores despite feature correlation.

## Key Findings

- The final model reaches **R² = 0.547 (log scale) / 0.552 (dollar scale)**, against 0.446 / 0.322 for an equivalent model given only specs, location, and host profile, price and engineered features materially improve the prediction.
- Gamma loss outperformed every alternative tested: squared-error loss (−0.067 R² log), Poisson loss (−0.037), linear regression (near zero R² log), and a tuned random forest (worse, and it overfit).
- Errors are not uniform across the revenue range: prediction accuracy is worst for the lowest-earning quintile of listings (~130% median error) and best for the highest-earning quintile (~38%).
- Amenities carry real signal individually, a single amenity count performed worse than giving each amenity (that isn't near-universal or near-nonexistent) its own binary flag.

## Limitations

1. The target itself is an estimate. Airbnb infers occupancy from review counts, so the model inherits whatever error that inference carries.
2. The model is least reliable for the lowest-earning listings.
3. Predictions are conditional on the price entered, the model estimates occupancy at that price, it does not recommend a price, and prices far outside the observed range are extrapolation.
4. Single snapshot (June 2026), no seasonality or event effects, a meaningful gap for a city with SXSW and ACL.
5. No listing-quality signals like photo quality or host responsiveness, since none of that is in the underlying data.

## Tech Stack

Python (pandas, NumPy, scikit-learn) · Google Colab · Cloudflare Workers (live demo deployment) · Claude (Sonnet 5)

## Repository Contents

- `airbnb_revenue_estimator.ipynb`: full notebook (preprocessing, feature engineering, model selection, evaluation, and an interactive prediction demo)
- `README.md`: this file
