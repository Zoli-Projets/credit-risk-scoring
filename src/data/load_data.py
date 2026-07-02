from pathlib import Path
import pandas as pd

# Chemin vers les données brutes
DATA_DIR = Path("data/raw")

def load_customers():
    """
    Load customers dataset.
    
    Returns:
        pd.DataFrame: Customers data
    """
    return pd.read_csv(DATA_DIR / "customers.csv")

def load_loans():
    """
    Load loans dataset.
    
    Returns:
        pd.DataFrame: Loans data
    """
    return pd.read_csv(DATA_DIR / "loans.csv")

def load_bureau():
    """
    Load bureau dataset.
    
    Returns:
        pd.DataFrame: Bureau data
    """
    return pd.read_csv(DATA_DIR / "bureau.csv")

def load_all():
    """
    Load all datasets.
    
    Returns:
        tuple: (customers_df, loans_df, bureau_df)
    """
    return load_customers(), load_loans(), load_bureau()