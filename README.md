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
├── .gitignore
├── setup.py
├── pyproject.toml
├── requirements.txt
├── README.md
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
└── data/
    ├── loans.csv
    ├── customers.csv
    └── bureau.csv
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

## Features

The preprocessing pipeline includes:

* Missing value handling
* Duplicate removal
* Categorical encoding
* Feature scaling
* Feature engineering
* Correlation analysis
* Variable selection using Information Value (IV)
* Multicollinearity analysis using VIF

Engineered features include:

* Loan-to-Income Ratio
* Delinquency Ratio
* Average Days Past Due per Delinquency

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

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* Matplotlib
* Seaborn

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
