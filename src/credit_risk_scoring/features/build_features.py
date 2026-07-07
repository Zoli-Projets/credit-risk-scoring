import pandas as pd
import numpy as np


def calculate_woe_iv(df, feature, target):
    """
    Calcule le WOE (Weight of Evidence) et l'IV (Information Value)
    d'une variable par rapport a une cible binaire.
    
    Parametres:
    - df: DataFrame contenant les donnees
    - feature: nom de la variable explicative
    - target: nom de la variable cible (0/1)
    
    Retourne:
    - grouped: DataFrame avec les statistiques par modalite
    - total_iv: Information Value total de la variable
    """
    # 1 Regrouper les donnees par modalite de la variable
    grouped = df.groupby(feature)[target].agg(['count', 'sum'])
    
    # 2 Renommer les colonnes
    grouped = grouped.rename(columns={
        'count': 'total',
        'sum': 'bad'
    })
    
    # 3 Calculer le nombre de "good" (non-defaillants)
    grouped['good'] = grouped['total'] - grouped['bad']
    
    # 4 Calculer les totaux globaux
    total_good = grouped['good'].sum()
    total_bad = grouped['bad'].sum()
    
    # 5 Calcul des proportions
    eps = 1e-6
    grouped['good_pct'] = (grouped['good'] + eps) / (total_good + eps)
    grouped['bad_pct'] = (grouped['bad'] + eps) / (total_bad + eps)
    
    # 6 Calcul du WOE
    grouped['woe'] = np.log(grouped['good_pct'] / grouped['bad_pct'])
    
    # 7 Calcul de l'IV par modalite
    grouped['iv'] = (grouped['good_pct'] - grouped['bad_pct']) * grouped['woe']
    
    # 8 Remplacer les valeurs infinies
    grouped['woe'] = grouped['woe'].replace([np.inf, -np.inf], 0)
    grouped['iv'] = grouped['iv'].replace([np.inf, -np.inf], 0)
    
    # 9 Calcul de l'IV total
    total_iv = grouped['iv'].sum()
    
    return grouped, total_iv


def compute_all_ivs(X_train, y_train):
    """
    Calcule l'Information Value pour toutes les variables.
    """
    y_train_int = y_train.astype(int)
    df_woe_full = pd.concat([X_train, y_train_int], axis=1)
    
    iv_values = {}
    
    for feature in X_train.columns:
        try:
            if X_train[feature].nunique() <= 2:
                # Variable binaire -> WOE direct
                _, iv = calculate_woe_iv(df_woe_full, feature, 'default')
            else:
                # Variable numerique -> discretisation en deciles
                X_binned = pd.qcut(
                    X_train[feature], q=10, labels=False, duplicates='drop'
                )
                df_binned = pd.concat(
                    [X_binned.rename(feature), y_train_int], axis=1
                )
                _, iv = calculate_woe_iv(df_binned, feature, 'default')
            
            iv_values[feature] = iv
            
        except Exception as e:
            print(f"Erreur sur {feature} : {e}")
            iv_values[feature] = 0
    
    return iv_values


def select_features_by_iv(iv_values, threshold=0.02):
    """
    Selectionne les variables dont l'IV est superieur au seuil.
    """
    selected_features = [
        feature for feature, iv in iv_values.items()
        if iv > threshold
    ]
    return selected_features