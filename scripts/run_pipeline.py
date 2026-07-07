import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import dump

# Ajouter le chemin racine pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credit_risk_scoring.data import (
    load_all_data, convert_currency, merge_data,
    preprocess_dataframe, split_train_test,
    clean_outliers, apply_business_rules,
    feature_engineering, prepare_for_modeling
)
from credit_risk_scoring.models import random_search_logistic

print("=" * 60)
print("CREDIT RISK SCORING - PIPELINE DE MODELISATION")
print("=" * 60)

# 1. Chargement des données
print("\n1. Chargement des données...")
loans, customers, bureau = load_all_data()
print(f"loans : {loans.shape}")
print(f"customers : {customers.shape}")
print(f"bureau : {bureau.shape}")

# 2. Conversion en euros
print("\n2. Conversion des unités monétaires en euros...")
customers, loans = convert_currency(customers, loans)

# 3. Fusion des tables
print("\n3. Fusion des bases de données...")
df = merge_data(customers, loans, bureau)
print(f"df fusionné : {df.shape}")

# 4. Prétraitement
print("\n4. Prétraitement des données...")
df = preprocess_dataframe(df)

# 5. Split Train/Test
print("\n5. Séparation Train/Test...")
df_train, df_test = split_train_test(df)
print(f"df_train : {df_train.shape}")
print(f"df_test : {df_test.shape}")

# 6. Nettoyage des outliers (processing_fee)
print("\n6. Nettoyage des valeurs aberrantes...")
df_train_1, df_test_1 = clean_outliers(df_train, df_test)
print(f"df_train_1 : {df_train_1.shape}")
print(f"df_test_1 : {df_test_1.shape}")

# 7. Application des règles de gestion
print("\n7. Application des règles de gestion métier...")
df_train_1, df_test_1 = apply_business_rules(df_train_1, df_test_1)
print(f"df_train après règles : {df_train_1.shape}")
print(f"df_test après règles : {df_test_1.shape}")

# 8. Feature Engineering
print("\n8. Feature Engineering...")
df_train_2, df_test_2 = feature_engineering(df_train_1, df_test_1)
print(f"df_train_2 : {df_train_2.shape}")
print(f"df_test_2 : {df_test_2.shape}")
print("Nouvelles colonnes créées :")
print([col for col in df_train_2.columns if col not in df_train_1.columns])

# 9. Préparation pour la modélisation
print("\n9. Préparation finale pour la modélisation...")
X_train, X_test, y_train, y_test, scaler, cols_to_scale = prepare_for_modeling(df_train_2, df_test_2)
print(f"X_train : {X_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"Features sélectionnées : {X_train.columns.tolist()}")

# 10. Entraînement du modèle final
print("\n10. Entraînement du modèle (Régression Logistique + RandomizedSearchCV)...")
best_model, best_params = random_search_logistic(X_train, y_train, X_test, y_test)

# 11. Sauvegarde du modèle
print("\n11. Sauvegarde du modèle...")
model_data = {
    'model': best_model,
    'features': X_train.columns,
    'scaler': scaler,
    'cols_to_scale': cols_to_scale
}
dump(model_data, 'model_data.joblib')
print("Modèle sauvegardé dans 'model_data.joblib'")

print("\n" + "=" * 60)
print("PIPELINE TERMINÉ AVEC SUCCÈS")
print("=" * 60)