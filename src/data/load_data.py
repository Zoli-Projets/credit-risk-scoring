from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path("data/raw")


def load_customers():
    return pd.read_csv(RAW_DATA_DIR / "customers.csv")


def load_loans():
    return pd.read_csv(RAW_DATA_DIR / "loans.csv")


def load_bureau():
    return pd.read_csv(RAW_DATA_DIR / "bureau.csv")


def load_all_data():
    """Load all raw datasets."""
    customers = load_customers()
    loans = load_loans()
    bureau = load_bureau()

    return customers, loans, bureau