# Repository: chess-winner-prediction-ml

## Project Title

Chess Winner Prediction Using Machine Learning

## Description

This project focuses on predicting the outcome of chess matches using historical game data and machine learning classification models. By analyzing features such as player ratings, move counts, and game outcomes, the project aims to build predictive models that assist in understanding key factors influencing victory in chess.

Built as part of a data science academic project, this solution applies classification algorithms such as K-Nearest Neighbors, Decision Tree, and XGBoost to derive accurate and interpretable insights from chess gameplay data.

## Objectives

- Preprocess and analyze historical chess match data
- Train multiple classification models to predict match outcomes
- Evaluate and compare model performance
- Explain model predictions using SHAP values

## Features

- Data cleaning and feature engineering for chess match datasets
- Implementation of classification models:
  - K-Nearest Neighbors
  - Decision Tree
  - XGBoost
- Model explainability using SHAP
- Performance evaluation using:
  - Accuracy
  - Precision, Recall, F1-Score
  - Confusion Matrix

## Technologies Used

| Tool/Library        | Purpose                                 |
|---------------------|-----------------------------------------|
| Python              | Core programming language               |
| pandas, NumPy       | Data processing and manipulation        |
| scikit-learn        | Machine learning models and utilities   |
| XGBoost             | Gradient boosting classification        |
| SHAP                | Model interpretability and explanation  |
| Matplotlib, Seaborn | Data visualization and evaluation       |
| Jupyter Notebook    | Interactive development environment     |

## Dataset

- **Source:** Lichess or Kaggle Chess Matches Dataset
- **Attributes Used:** Player Elo ratings, number of moves, game result, time controls, opening moves, etc.
- **Size:** ~20,000+ match records

## File Structure


## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Sravansai-2001/chess-winner-prediction-ml.git
cd chess-winner-prediction-ml
pip install -r requirements.txt
jupyter notebook notebooks/chess_prediction_model.ipynb
