import pandas as pd
import numpy as np
from typing import Optional, List
from src.utils.config import RANDOM_STATE


class DataPreprocessor:
    """
    Data cleaning pipeline for credit risk data.
    
    This class implements a chainable preprocessing pipeline with methods
    for handling missing values, outliers, encoding, and scaling.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize preprocessor with a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame to process
        """
        self.df = df.copy()
        self._original_shape = df.shape
    
    def clean_missing_values(self, strategy: str = 'median', threshold: float = 0.5) -> 'DataPreprocessor':
        """
        Handle missing values in the dataset.
        
        Args:
            strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
            threshold (float): Maximum proportion of missing values allowed per column
            
        Returns:
            DataPreprocessor: Self for method chaining
        """
        # Supprimer les colonnes avec trop de valeurs manquantes
        missing_ratio = self.df.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            print(f"🗑️ Dropped columns with >{threshold*100}% missing: {cols_to_drop}")
        
        # Imputer les valeurs manquantes restantes
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if self.df[col].dtype in ['float64', 'int64']:
                    if strategy == 'mean':
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    elif strategy == 'median':
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    else:
                        self.df[col].fillna(0, inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown', inplace=True)
        
        return self
    
    def remove_outliers(self, method: str = 'iqr', factor: float = 1.5) -> 'DataPreprocessor':
        """
        Remove outliers from numerical columns.
        
        Args:
            method (str): Outlier detection method ('iqr', 'zscore')
            factor (float): Multiplier for IQR or Z-score threshold
            
        Returns:
            DataPreprocessor: Self for method chaining
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                self.df = self.df[z_scores <= factor]
        
        return self
    
    def validate_business_rules(self) -> 'DataPreprocessor':
        """
        Apply business validation rules specific to credit risk.
        
        Common rules:
        - Age must be >= 18
        - Loan amount must be positive
        - Income must be positive
        
        Returns:
            DataPreprocessor: Self for method chaining
        """
        # Règles métier à adapter selon votre dataset
        if 'age' in self.df.columns:
            self.df = self.df[self.df['age'] >= 18]
            self.df = self.df[self.df['age'] <= 100]
        
        if 'loan_amount' in self.df.columns:
            self.df = self.df[self.df['loan_amount'] > 0]
        
        if 'income' in self.df.columns:
            self.df = self.df[self.df['income'] > 0]
        
        return self
    
    def encode_categories(self, method: str = 'one_hot') -> 'DataPreprocessor':
        """
        Encode categorical variables.
        
        Args:
            method (str): Encoding method ('one_hot', 'label')
            
        Returns:
            DataPreprocessor: Self for method chaining
        """
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        if method == 'one_hot':
            self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in categorical_cols:
                self.df[col] = le.fit_transform(self.df[col].astype(str))
        
        return self
    
    def scale_features(self, method: str = 'standard') -> 'DataPreprocessor':
        """
        Scale numerical features.
        
        Args:
            method (str): Scaling method ('standard', 'minmax')
            
        Returns:
            DataPreprocessor: Self for method chaining
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            return self
        
        self.df[numerical_cols] = scaler.fit_transform(self.df[numerical_cols])
        
        return self
    
    def transform(self) -> pd.DataFrame:
        """
        Return the processed DataFrame.
        
        Returns:
            pd.DataFrame: Cleaned and preprocessed DataFrame
        """
        print(f"✅ Preprocessing complete: {self._original_shape[0]} → {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def get_summary(self) -> dict:
        """
        Get summary statistics of the processing.
        
        Returns:
            dict: Summary dictionary
        """
        return {
            'original_rows': self._original_shape[0],
            'original_cols': self._original_shape[1],
            'final_rows': self.df.shape[0],
            'final_cols': self.df.shape[1],
            'missing_after': self.df.isnull().sum().sum()
        }