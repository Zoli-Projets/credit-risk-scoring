import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc
from joblib import load

# Ajouter le chemin parent pour importer les modules src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_data import load_all_data
from src.data.preprocess import (
    convert_currency, merge_data, preprocess_dataframe,
    split_train_test, clean_outliers, apply_business_rules,
    feature_engineering, prepare_for_modeling
)

print("=" * 60)
print("CREDIT RISK SCORING - ÉVALUATION DU MODÈLE")
print("=" * 60)

# 1. Chargement et préparation des données (identique au pipeline)
print("\n1. Chargement et préparation des données...")
loans, customers, bureau = load_all_data()
customers, loans = convert_currency(customers, loans)
df = merge_data(customers, loans, bureau)
df = preprocess_dataframe(df)
df_train, df_test = split_train_test(df)
df_train_1, df_test_1 = clean_outliers(df_train, df_test)
df_train_1, df_test_1 = apply_business_rules(df_train_1, df_test_1)
df_train_2, df_test_2 = feature_engineering(df_train_1, df_test_1)
X_train, X_test, y_train, y_test, scaler, cols_to_scale = prepare_for_modeling(df_train_2, df_test_2)

# 2. Chargement du modèle sauvegardé
print("\n2. Chargement du modèle...")
artifacts = load('model_data.joblib')
model = artifacts['model']
features = artifacts['features']
scaler = artifacts['scaler']
cols_to_scale = artifacts['cols_to_scale']

# 3. Préparation des données de test avec le bon scaler
print("\n3. Préparation des données de test...")
X_test_final = X_test[features].copy()
X_test_final[cols_to_scale] = scaler.transform(X_test_final[cols_to_scale])

# 4. Prédiction et classification report
print("\n4. Évaluation du modèle...")
y_pred = model.predict(X_test_final)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# 5. ROC Curve et AUC
print("\n5. Calcul de la courbe ROC et AUC...")
probabilities = model.predict_proba(X_test_final)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
area = auc(fpr, tpr)
print(f"AUC : {area:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {area:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Modèle aléatoire (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.title('Courbe ROC — Régression Logistique + RandomizedSearchCV')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.show()

# 6. Coefficient de Gini
print("\n6. Calcul du Coefficient de Gini...")
gini_coefficient = 2 * area - 1
print(f"Gini Coefficient : {gini_coefficient:.4f}")
print("Interprétation :")
print("- < 0.20 : modèle faible")
print("- 0.20–0.40 : modèle acceptable")
print("- 0.40–0.60 : bon modèle")
print("- > 0.60 : excellent modèle")

# 7. Rank Ordering et KS Statistic
print("\n7. Rank Ordering et KS Statistic...")

decile_df = pd.DataFrame({
    'y_test': y_test.values,
    'probability': probabilities
})

decile_df['decile'] = pd.qcut(
    decile_df['probability'], q=10, labels=False, duplicates='drop'
)
decile_df['decile'] = 9 - decile_df['decile']

rank_order = decile_df.groupby('decile').agg(
    total=('y_test', 'count'),
    events=('y_test', 'sum'),
).reset_index()

rank_order['non_events'] = rank_order['total'] - rank_order['events']
rank_order['event_rate'] = (rank_order['events'] / rank_order['total']).round(4)
rank_order['cum_events'] = rank_order['events'].cumsum()
rank_order['cum_non_events'] = rank_order['non_events'].cumsum()
rank_order['cum_event_pct'] = (rank_order['cum_events'] / rank_order['events'].sum()).round(4)
rank_order['cum_non_event_pct'] = (rank_order['cum_non_events'] / rank_order['non_events'].sum()).round(4)
rank_order['KS'] = (rank_order['cum_event_pct'] - rank_order['cum_non_event_pct']).abs().round(4)

print("\n--- Table des déciles — Rank Ordering ---")
print(rank_order.to_string(index=False))

ks_stat = rank_order['KS'].max()
ks_decile = rank_order.loc[rank_order['KS'].idxmax(), 'decile']
print(f"\nKS Statistic  : {ks_stat:.4f} ({ks_stat*100:.2f}%)")
print(f"Décile optimal : {ks_decile}")

print("\nInterprétation KS :")
print("- < 0.20 : modèle faible")
print("- 0.20 – 0.40 : modèle acceptable")
print("- 0.40 – 0.60 : bon modèle")
print("- > 0.60 : excellent modèle")

# Visualisation KS
plt.figure(figsize=(10, 5))
plt.plot(rank_order['decile'], rank_order['cum_event_pct'],
         marker='o', label='Événements cumulés (défauts)', color='red')
plt.plot(rank_order['decile'], rank_order['cum_non_event_pct'],
         marker='o', label='Non-événements cumulés', color='blue')
plt.bar(rank_order['decile'], rank_order['KS'],
        alpha=0.3, color='green', label='KS par décile')
plt.axvline(x=ks_decile, color='green', linestyle='--',
            label=f'KS max = {ks_stat:.2%} au décile {ks_decile}')
plt.xlabel('Décile')
plt.ylabel('Taux cumulé')
plt.title('KS Statistic — Rank Ordering')
plt.legend()
plt.tight_layout()
plt.savefig('ks_statistic.png')
plt.show()

# 8. Importance des variables
print("\n8. Importance des variables...")
feature_importance = model.coef_[0]
coef_df = pd.DataFrame(
    feature_importance,
    index=X_train.columns,
    columns=['Coefficients']
).sort_values(by='Coefficients', ascending=True)

print("\n--- Top 5 variables augmentant le risque (coefficients positifs) ---")
print(coef_df.tail(5))
print("\n--- Top 5 variables réduisant le risque (coefficients négatifs) ---")
print(coef_df.head(5))

plt.figure(figsize=(8, 4))
plt.barh(coef_df.index, coef_df['Coefficients'], color='steelblue')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel('Valeur du coefficient')
plt.title('Importance des variables — Régression Logistique + RandomizedSearchCV')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("\n" + "=" * 60)
print("ÉVALUATION TERMINÉE")
print("=" * 60)