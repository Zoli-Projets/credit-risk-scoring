# Credit Risk Scoring

## Overview

This project was developed as part of a Master's degree in Data Science. Its objective is to predict whether a borrower will default on a loan using demographic, financial, and credit history information.

The project follows a complete machine learning workflow, including:
- Data preprocessing and cleaning
- Feature engineering
- Exploratory Data Analysis (EDA)
- Model training and hyperparameter optimization
- In-depth evaluation and interpretation

**AUC : 0.9835** | **Gini : 0.9671** | **KS : 85.48%**
---

## Project Structure

```text
credit-risk-scoring/
├──config
│  └── config.yaml
│
├── requirements.txt
│
│── data/
│    ├── loans.csv
│    ├── customers.csv
│    └── bureau.csv
│
├── reports/
│   └── Rapport_projet.pdf
│
├── notebooks/
│   └── notebook.ipynb
│
├── scripts/
│   ├── run_pipeline.py
│   └── evaluate_model.py
│
├── src/
│   └── credit_risk_scoring/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── load_data.py
│       │   └── preprocess.py
│       ├── features/
│       │   ├── __init__.py
│       │   └── build_features.py
│       └── models/
│           ├── __init__.py
│           └── train_model.py
│
├── .gitignore
├── pyproject.toml
├── README.md
└── setup.py    
```

---

## Dataset

The project uses three datasets:

* **customers.csv** – Customer demographic information (age, gender, income, employment status, etc.)

* **loans.csv** – Loan characteristics (amount, purpose, type, fees, disbursal dates, etc.)

* **bureau.csv** – Credit bureau history (number of accounts, delinquencies, credit utilization, etc.)
Dataset
 

These datasets are merged to build the final modeling dataset.

> **Note:** The `data/` folder is not intended to be version-controlled. Place the datasets manually in this directory before running the project.

---
Methodology

The preprocessing pipeline includes:

1. Data Preprocessing
Currency conversion: All monetary values converted from INR to EUR

Missing value handling: Imputation of missing values

Duplicate removal: No duplicates found in the dataset

Outlier detection: Removal of unrealistic values (processing_fee > 3% of loan_amount)

Business rules validation: GST ≤ 20% of loan_amount, net_disbursement ≤ loan_amount

Categorical encoding: One-Hot Encoding of categorical variables

### 2. Feature Engineering
Engineered features include:

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| **Loan-to-Income Ratio (LTI)** | `loan_amount / income` | Measures relative debt burden |
| **Delinquency Ratio** | `(delinquent_months / total_loan_months) × 100` | Proportion of time in delinquency |
| **Average DPD** | `total_dpd / delinquent_months` | Intensity of payment delays |
 
3. Feature Selection
VIF (Variance Inflation Factor): Detection and removal of multicollinearity

IV (Information Value): Selection of most predictive variables (IV > 0.02)

4. Modeling & Optimization
Models tested: Logistic Regression, Random Forest, XGBoost

Optimization: RandomizedSearchCV and Optuna for hyperparameter tuning

Imbalance handling: Random Under-Sampling and SMOTE Tomek

5. Evaluation
AUC / ROC : Discrimination power

Gini Coefficient : Ranking ability

KS Statistic : Class separation

Rank Ordering : Risk segmentation



---

## Models

Several machine learning algorithms were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* LightGBM
* CatBoost

Hyperparameter optimization was performed to improve model performance.

---

## Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Precision-Recall AUC
* Confusion Matrix

Special attention is given to **Recall** for the default class, as minimizing false negatives is crucial in credit risk prediction.

---

### Final Model: Logistic Regression with RandomizedSearchCV

| Métrique | Valeur |
|----------|--------|
| **AUC** | 0.9835 |
| **Coefficient de Gini** | 0.9671 |
| **KS Statistic** | 85.48% |
| **Accuracy** | 96% |
| **F1-score (classe défaut)** | 0.78 |
| **Recall (classe défaut)** | 0.73 |

---

### Variable Importance (Top 5)

| Variable | Coefficient | Impact |
|----------|-------------|--------|
| `loan_to_income` | 15.824 | ⬆️⬆️⬆️ Risque |
| `credit_utilization_ratio` | 14.727 | ⬆️⬆️⬆️ Risque |
| `delinquency_ratio` | 10.960 | ⬆️⬆️⬆️ Risque |
| `avg_dpd_per_delinquency` | 2.072 | ⬆️⬆️ Risque |
| `residence_type_Rented` | 1.604 | ⬆️ Risque |

---

### Decile Analysis

| Décile | Taux d'Événements | Taux de Non-Événements | Cumul Événements |
|--------|-------------------|------------------------|------------------|
| 9 (risque élevé) | 71.92% | 28.08% | 83.70% |
| 8 | 12.88% | 87.12% | 98.70% |
| 7 | 0.72% | 99.28% | 99.53% |
| 6 | 0.40% | 99.60% | 99.93% |
| 5 à 0 | 0.00% | 100.00% | 100.00% |

> ✅ **Interpretation:** The first two deciles concentrate nearly 99% of defaults, confirming the model's excellent risk segmentation capability.

## Installation

Clone the repository:

```bash
git clone https://github.com/Zoli-Projets/credit-risk-scoring.git

cd credit-risk-scoring
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Or install the package in editable mode:

```bash
pip install -e .
```

---

## Running the Project

Run the complete pipeline:

```bash
python scripts/run_pipeline.py
```

Evaluate the trained model:

```bash
python scripts/evaluate_model.py
```

---

## Exploratory Data Analysis

A complete exploratory analysis is available in:

```text
notebooks/notebook.ipynb
```

---
## Technologies

### Langage et environnement

- **Python 3.9+** : Langage principal du projet

### Bibliothèques principales
 
| Catégorie | Bibliothèques |
|-----------|---------------|
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM, CatBoost |
| **Optimization** | Optuna |
| **Visualization** | Matplotlib, Seaborn |
| **Statistics** | Statsmodels |
| **Model Persistence** | Joblib |
| **Imbalance Handling** | Imbalanced-learn |
---

## Results

The project compares several machine learning models for credit default prediction and identifies the best-performing model after hyperparameter tuning.

The complete methodology and experimental results are available in:

```text
reports/Rapport_projet.pdf
```

---

## License

This project is licensed under the MIT License.
