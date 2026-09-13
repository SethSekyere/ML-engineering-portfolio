# CardioRisk — Heart Disease Classification with Tree Ensembles

A machine learning project comparing **Decision Trees, Random Forests, and XGBoost** for binary heart disease classification using structured clinical data.

## Overview

The project explores how different tree-based algorithms perform as model complexity increases, including:

* Decision Tree hyperparameter tuning
* Random Forest ensemble learning
* XGBoost gradient boosting
* Early stopping
* Model performance comparison

## Tech Stack

* Python
* NumPy
* Pandas
* Scikit-learn
* XGBoost
* Matplotlib
* Google Colab

## Models

| Model         | Key Concepts                                     |
| ------------- | ------------------------------------------------ |
| Decision Tree | Tree depth, minimum samples                      |
| Random Forest | Ensemble learning, number of trees               |
| XGBoost       | Gradient boosting, learning rate, early stopping |

## Dataset

The project uses the **Heart Failure Prediction Dataset** with demographic, clinical, and exercise-related features. Categorical features are one-hot encoded before model training.
If you decide to run this project in your own environment, kindly download the dataset and save it as "heart csv"

## Key Takeaway

The project demonstrates the progression from a single decision tree to ensemble-based methods and highlights the importance of **hyperparameter tuning, validation, and controlling model complexity**.

> **Note:** This project is for educational and portfolio purposes and is not intended for clinical diagnosis or medical decision-making.

## Project Structure

```text
01-CardioRisk-Tree-Ensembles/
├── Project-README.md
├── CardioRisk-Tree-Ensembles.ipynb
└── deeplearning.mplstyle
```
