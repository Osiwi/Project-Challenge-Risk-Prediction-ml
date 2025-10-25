# Project Challenge Risk Prediction (ML)

This repository contains a machine learning workflow to predict whether a project will encounter delivery challenges ("yes" vs "no"). The goal is early risk detection, so teams can intervene before failure.

## What this model does
- Cleans and preprocesses project management data (numeric + categorical features)
- Handles missing values
- Scales numerical features
- One-hot encodes categorical features
- Uses SMOTE to fix class imbalance so the model learns from the minority ("challenge = yes") class instead of ignoring it
- Trains and compares multiple classifiers:
  - Random Forest
  - Gradient Boosting
  - MLPClassifier (neural network)
  - AdaBoost
  - Bagging
  - LightGBM
  - XGBoost
- Uses Bayesian hyperparameter optimization with cross-validation to tune models
- Evaluates using:
  - F1-score
  - ROC AUC
  - Precision–Recall curve
  - Confusion matrix

## Notebook
All the core code and experiments live in:

`notebooks/project_challenge_prediction_modeling.ipynb`

This notebook:
1. Builds the preprocessing pipeline
2. Applies SMOTE resampling to address class imbalance
3. Trains/tunes multiple models
4. Compares their performance on hold-out test data

## Why this matters
Most project tracking is reactive. This workflow makes it predictive:
- Instead of “a problem happened,” we ask “will this project become a problem?”
- That supports proactive resource allocation, escalation, and governance.

## Related work
- Upstream analytics and data engineering (energy, emissions, and operational context) are handled in a separate pipeline repository.
  This ML model can plug into that type of operational dataset to generate early warning risk signals.
