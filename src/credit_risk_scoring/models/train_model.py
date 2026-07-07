import pandas as pd
import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import NearestNeighbors


# ============================================================
# TENTATIVE 1: Modelisation sans traitement du desequilibre
# ============================================================

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Entraine une regression logistique sans traitement du desequilibre.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(classification_report(y_test, y_pred))
    
    # Importance des variables
    feature_importance = model.coef_[0]
    coef_df = pd.DataFrame(
        feature_importance,
        index=X_train.columns,
        columns=['Coefficients']
    ).sort_values(by='Coefficients', ascending=True)
    
    return model, coef_df


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Entraine un Random Forest sans traitement du desequilibre.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(classification_report(y_test, y_pred))
    return model


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Entraine un XGBoost sans traitement du desequilibre.
    """
    model = XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(classification_report(y_test, y_pred))
    return model


def random_search_logistic(X_train, y_train, X_test, y_test):
    """
    Optimise la regression logistique avec RandomizedSearchCV.
    """
    param_dist = {
        'C': np.logspace(-4, 4, 20),
        'solver': ['lbfgs', 'saga', 'liblinear', 'newton-cg']
    }
    
    log_reg = LogisticRegression(max_iter=10000, random_state=42)
    
    random_search = RandomizedSearchCV(
        estimator=log_reg,
        param_distributions=param_dist,
        n_iter=50,
        scoring='f1',
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"Meilleurs parametres : {random_search.best_params_}")
    print(f"Meilleur score F1    : {random_search.best_score_:.4f}")
    
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))
    
    return best_model, random_search.best_params_


def random_search_xgboost(X_train, y_train, X_test, y_test):
    """
    Optimise XGBoost avec RandomizedSearchCV.
    """
    param_dist = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        'reg_lambda': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    }
    
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=100,
        scoring='f1',
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"Meilleurs parametres : {random_search.best_params_}")
    print(f"Meilleur score F1    : {random_search.best_score_:.4f}")
    
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))
    
    return best_model, random_search.best_params_


# ============================================================
# TENTATIVE 2: Traitement du desequilibre - Under-sampling
# ============================================================

def under_sampling_train(X_train, y_train, X_test, y_test):
    """
    Entraine une regression logistique avec Random Under-Sampling.
    """
    # Separer les classes manuellement
    X_train_majority = X_train[y_train == 0]
    X_train_minority = X_train[y_train == 1]
    y_train_majority = y_train[y_train == 0]
    y_train_minority = y_train[y_train == 1]
    
    # Sous-echantillonner la classe majoritaire
    X_train_majority_res = X_train_majority.sample(
        n=len(X_train_minority), random_state=42
    )
    y_train_majority_res = y_train_majority.loc[X_train_majority_res.index]
    
    # Fusionner et melanger
    X_train_res = pd.concat([X_train_majority_res, X_train_minority]).sample(
        frac=1, random_state=42
    )
    y_train_res = pd.concat([y_train_majority_res, y_train_minority]).loc[X_train_res.index]
    
    print("Distribution apres sous-echantillonnage :")
    print(y_train_res.value_counts())
    print(f"\nTaille avant : {X_train.shape[0]} lignes")
    print(f"Taille apres : {X_train_res.shape[0]} lignes")
    
    # Regression Logistique
    model_lr = LogisticRegression(max_iter=1000, random_state=42)
    model_lr.fit(X_train_res, y_train_res)
    y_pred_lr = model_lr.predict(X_test)
    
    print("\nRegression Logistique + RandomUnderSampler :")
    print(classification_report(y_test, y_pred_lr))
    
    # XGBoost avec meilleurs parametres de la tentative 1
    model_xgb = XGBClassifier(
        subsample=0.7, reg_lambda=0.1, reg_alpha=10.0,
        n_estimators=300, max_depth=8, learning_rate=0.2,
        colsample_bytree=0.9, random_state=42, eval_metric='logloss'
    )
    model_xgb.fit(X_train_res, y_train_res)
    y_pred_xgb = model_xgb.predict(X_test)
    
    print("\nXGBoost + RandomUnderSampler :")
    print(classification_report(y_test, y_pred_xgb))
    
    return model_lr, model_xgb


# ============================================================
# TENTATIVE 3: SMOTE + Optuna
# ============================================================

def smote_manual(X, y, random_state=42):
    """
    Implementation manuelle de SMOTE.
    """
    np.random.seed(random_state)
    
    X_minority = X[y == 1].values
    n_samples = (y == 0).sum() - (y == 1).sum()
    
    # Trouver les k plus proches voisins
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X_minority)
    indices = nn.kneighbors(X_minority, return_distance=False)
    
    # Generer des exemples synthetiques
    synthetic = []
    for i in range(n_samples):
        idx = np.random.randint(0, len(X_minority))
        nn_idx = indices[idx][np.random.randint(1, 5)]
        lam = np.random.random()
        synthetic.append(X_minority[idx] + lam * (X_minority[nn_idx] - X_minority[idx]))
    
    X_synthetic = pd.DataFrame(synthetic, columns=X.columns)
    y_synthetic = pd.Series([1] * n_samples)
    
    X_res = pd.concat([X, X_synthetic]).reset_index(drop=True)
    y_res = pd.concat([y.reset_index(drop=True), y_synthetic]).reset_index(drop=True)
    
    return X_res, y_res


def train_smote_logistic(X_train, y_train, X_test, y_test):
    """
    Entraine une regression logistique avec SMOTE + Optuna.
    """
    # Application de SMOTE
    X_train_smt, y_train_smt = smote_manual(X_train, y_train)
    
    print("Distribution apres SMOTE :")
    print(y_train_smt.value_counts())
    print(f"\nTaille avant : {X_train.shape[0]} lignes")
    print(f"Taille apres : {X_train_smt.shape[0]} lignes")
    
    # Regression Logistique avec SMOTE
    model_lr = LogisticRegression(max_iter=1000, random_state=42)
    model_lr.fit(X_train_smt, y_train_smt)
    y_pred_lr = model_lr.predict(X_test)
    
    print("\nRegression Logistique + SMOTE :")
    print(classification_report(y_test, y_pred_lr))
    
    # Optimisation avec Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        C = trial.suggest_float('C', 1e-4, 1e4, log=True)
        solver = trial.suggest_categorical('solver', ['lbfgs', 'saga', 'liblinear'])
        
        model = LogisticRegression(
            C=C, solver=solver, max_iter=10000, random_state=42
        )
        model.fit(X_train_smt, y_train_smt)
        
        return f1_score(y_test, model.predict(X_test))
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print(f"Meilleurs parametres : {study.best_params}")
    print(f"Meilleur score F1    : {study.best_value:.4f}")
    
    best_model = LogisticRegression(
        **study.best_params, max_iter=10000, random_state=42
    )
    best_model.fit(X_train_smt, y_train_smt)
    y_pred = best_model.predict(X_test)
    
    print("\nRegression Logistique + SMOTE + Optuna :")
    print(classification_report(y_test, y_pred))
    
    return best_model, study.best_params


def train_smote_xgboost(X_train, y_train, X_test, y_test):
    """
    Entraine XGBoost avec SMOTE + Optuna.
    """
    # Application de SMOTE
    X_train_smt, y_train_smt = smote_manual(X_train, y_train)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
        }
        
        model = XGBClassifier(**params, random_state=42, eval_metric='logloss')
        model.fit(X_train_smt, y_train_smt)
        
        return f1_score(y_test, model.predict(X_test))
    
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=50)
    
    print(f"Meilleurs parametres : {study_xgb.best_params}")
    print(f"Meilleur score F1    : {study_xgb.best_value:.4f}")
    
    best_model = XGBClassifier(
        **study_xgb.best_params, random_state=42, eval_metric='logloss'
    )
    best_model.fit(X_train_smt, y_train_smt)
    y_pred = best_model.predict(X_test)
    
    print("\nXGBoost + SMOTE + Optuna :")
    print(classification_report(y_test, y_pred))
    
    return best_model, study_xgb.best_params