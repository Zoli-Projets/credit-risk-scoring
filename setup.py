from setuptools import setup, find_packages

setup(
    name="credit-risk-scoring",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.5.0",
        "xgboost>=2.1.0",
        "imbalanced-learn>=0.12.0",
        "optuna>=4.0.0",
        "statsmodels>=0.14.0",
        "joblib>=1.4.0",
        "seaborn>=0.13.0",
        "matplotlib>=3.8.0",
    ],
)