
# CardioRisk — Heart Disease Risk Classification with Tree Ensembles

A machine learning classification project that evaluates **Decision Trees, Random Forests, and XGBoost** for identifying patterns associated with heart disease from patient-level clinical features.

The project focuses on understanding how increasingly sophisticated tree-based methods improve predictive performance, how model complexity affects generalization, and how hyperparameter tuning and early stopping can be used to control overfitting.

> **Note:** This is an educational machine learning project and is **not intended for clinical diagnosis or medical decision-making**.

---

## Project Overview

CardioRisk explores a common supervised learning problem: predicting a binary heart disease outcome from structured clinical data.

Rather than relying on a single model, the project builds a progression of tree-based classifiers:

**Decision Tree → Random Forest → XGBoost**

This progression provides a practical comparison between a single interpretable decision tree and increasingly powerful ensemble-learning approaches.

The project investigates:

* How tree depth affects model complexity
* The impact of minimum sample requirements on generalization
* How ensemble methods improve upon individual decision trees
* The effect of the number of trees in a Random Forest
* Gradient boosting with XGBoost
* Early stopping to limit unnecessary boosting iterations
* Model performance on held-out data

---

## Dataset

The project uses the **Heart Failure Prediction Dataset** containing clinical and demographic variables associated with heart disease.

The target variable is:

* `HeartDisease` — binary classification target

Features include:

| Feature          | Description                           |
| ---------------- | ------------------------------------- |
| `Age`            | Patient age                           |
| `Sex`            | Patient sex                           |
| `ChestPainType`  | Type of chest pain                    |
| `RestingBP`      | Resting blood pressure                |
| `Cholesterol`    | Cholesterol measurement               |
| `FastingBS`      | Fasting blood sugar indicator         |
| `RestingECG`     | Resting electrocardiogram result      |
| `MaxHR`          | Maximum heart rate achieved           |
| `ExerciseAngina` | Exercise-induced angina indicator     |
| `Oldpeak`        | ST depression measurement             |
| `ST_Slope`       | Slope of the peak exercise ST segment |

Categorical variables are transformed using **one-hot encoding** before being passed to the machine learning models.

---

## Machine Learning Pipeline

The workflow follows a standard supervised learning pipeline:

```text
Clinical Dataset
       │
       ▼
Data Inspection
       │
       ▼
Categorical Encoding
       │
       ▼
Train / Test Split
       │
       ├───────────────┐
       ▼               ▼
Decision Tree      Ensemble Models
       │               │
       │        ┌──────┴──────┐
       │        ▼             ▼
       │   Random Forest   XGBoost
       │
       └───────────────┬───────
                       ▼
                Model Evaluation
```

The dataset is split into training and held-out evaluation data using a fixed random state to improve reproducibility.

---

## Models

### 1. Decision Tree

A Decision Tree provides the baseline model.

The project explores tree complexity by varying parameters such as:

* `max_depth`
* `min_samples_split`

This demonstrates the tradeoff between model flexibility and generalization.

A shallow tree may underfit the data, while an excessively deep tree can memorize training patterns.

---

### 2. Random Forest

Random Forest extends the decision-tree approach by training an ensemble of trees and aggregating their predictions.

The project investigates parameters including:

* `n_estimators`
* `max_depth`
* `min_samples_split`

This provides a useful comparison between a single tree and an ensemble of independently trained trees.

The final Random Forest configuration uses:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=16,
    min_samples_split=10,
    random_state=RANDOM_STATE
)
```

---

### 3. XGBoost

The final model uses **XGBoost**, a gradient-boosted decision-tree algorithm.

Unlike Random Forest, where trees are trained independently, gradient boosting builds trees sequentially, with each new tree attempting to improve upon the errors of the existing ensemble.

The project uses:

* `n_estimators=500`
* `learning_rate=0.1`
* `early_stopping_rounds=10`

Early stopping monitors performance on a validation set and prevents the boosting process from continuing unnecessarily once additional trees stop providing improvement.

---

## Model Development Strategy

The project intentionally builds the models incrementally rather than jumping directly to a complex algorithm.

### Baseline

A Decision Tree establishes an initial reference point.

### Hyperparameter Exploration

Tree depth and minimum sample requirements are varied to examine their effect on classification performance.

### Ensemble Learning

Random Forest is introduced to determine whether combining multiple decision trees provides more robust predictions.

### Gradient Boosting

XGBoost is then evaluated as a more sequential and optimization-focused ensemble method.

### Early Stopping

A validation subset is used during XGBoost training to determine when additional boosting iterations no longer improve the model.

---

## Technologies

* **Python**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **XGBoost**
* **Matplotlib**
* **Jupyter Notebook / Google Colab**

---

## Key ML Concepts Demonstrated

This project demonstrates practical understanding of:

* Supervised learning
* Binary classification
* Categorical feature encoding
* Train/test data splitting
* Decision tree learning
* Ensemble learning
* Random Forests
* Gradient boosting
* XGBoost
* Hyperparameter tuning
* Model complexity
* Overfitting vs. underfitting
* Validation-based early stopping
* Reproducible experimentation

---

## Results

The project compares the classification performance of:

| Model         | Approach                       |
| ------------- | ------------------------------ |
| Decision Tree | Single tree                    |
| Random Forest | Bagged tree ensemble           |
| XGBoost       | Gradient-boosted tree ensemble |

The experiments demonstrate how moving from a single decision tree to ensemble methods can provide more robust classification performance.

The notebook contains the detailed experiments, parameter sweeps, visualizations, and model evaluation.

---

## Project Structure

```text
01-CardioRisk-Tree-Ensembles/
│
├── README.md
├── CardioRisk-Tree-Ensembles.ipynb
└── deeplearning.mplstyle
```

---

## Reproducibility

The notebook is designed to run in a Google Colab environment.

The project uses a fixed random state:

```python
RANDOM_STATE = 55
```

to make the train/test split and model experiments reproducible.

The dataset is **not intended to remain in the GitHub repository**. The notebook should instead provide clear instructions for obtaining the original dataset before execution.

---

## Why This Project Matters

Tree-based models are particularly useful for structured/tabular data because they can model nonlinear relationships and interactions between features without requiring feature scaling.

This project demonstrates the progression from a simple interpretable model to modern ensemble techniques while emphasizing an important ML engineering principle:

> **Model selection should be driven by experimentation and evaluation rather than assuming that a more complex algorithm will automatically perform better.**

---

## Future Improvements

Potential extensions include:

* Add precision, recall, F1-score, and ROC-AUC alongside accuracy
* Generate confusion matrices for each model
* Compare feature importance across models
* Investigate class imbalance
* Perform systematic cross-validation
* Use more rigorous hyperparameter search
* Add model calibration analysis
* Track experiments and metrics programmatically
* Package the final model behind a lightweight prediction API

---

## Disclaimer

This project is intended solely for **educational and portfolio purposes**. The model should not be used to diagnose heart disease, make clinical decisions, or replace evaluation by qualified healthcare professionals.
