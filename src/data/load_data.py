from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")


def load_loans():

    return pd.read_csv(DATA_DIR / "loans.csv")


def load_customers():

    return pd.read_csv(DATA_DIR / "customers.csv")


def load_bureau():

    return pd.read_csv(DATA_DIR / "bureau_data.csv")


def load_all_data():

    loans = load_loans()

    customers = load_customers()

    bureau = load_bureau()

    return loans, customers, bureau