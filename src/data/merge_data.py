import pandas as pd
from pathlib import Path
from .load_data import load_customers, load_loans, load_bureau

def merge_customer_loans(customers_df: pd.DataFrame, loans_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge customer and loan data.
    
    Args:
        customers_df (pd.DataFrame): Customer data
        loans_df (pd.DataFrame): Loan data
        
    Returns:
        pd.DataFrame: Merged dataset
    """
    # Fusion sur l'ID client (à adapter selon votre schéma de données)
    merged = pd.merge(customers_df, loans_df, on='customer_id', how='inner')
    return merged

def merge_all_data() -> pd.DataFrame:
    """
    Load and merge all datasets.
    
    Returns:
        pd.DataFrame: Complete merged dataset
    """
    customers, loans, bureau = load_customers(), load_loans(), load_bureau()
    
    # Premier merge
    merged = merge_customer_loans(customers, loans)
    
    # Deuxième merge avec bureau (à adapter)
    merged = pd.merge(merged, bureau, on='customer_id', how='left')
    
    return merged

def save_merged_data(merged_df: pd.DataFrame, output_path: str = None):
    """
    Save merged data to CSV.
    
    Args:
        merged_df (pd.DataFrame): Data to save
        output_path (str, optional): Output path. Defaults to None.
    """
    from src.utils.config import PROCESSED_DATA
    
    if output_path is None:
        output_path = PROCESSED_DATA / "merged_data.csv"
    
    merged_df.to_csv(output_path, index=False)
    print(f"✅ Data saved to {output_path}")