from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

def cross_validate(

    model,

    X,

    y

):

    cv = StratifiedKFold(

        n_splits=5,

        shuffle=True,

        random_state=42

    )

    auc = cross_val_score(

        model,

        X,

        y,

        cv=cv,

        scoring="roc_auc"

    )

    return auc