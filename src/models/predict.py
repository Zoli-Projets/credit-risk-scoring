def train_all_models(

    X_train,

    y_train,

    X_test,

    y_test

):

    models={

        "Logistic":LogisticModel(),

        "Random Forest":RandomForestModel(),

        "XGBoost":XGBoostModel()

    }

    results={}

    best_model=None

    best_auc=0

    for name,model in models.items():

        print(f"\nTraining {name}")

        model.train(X_train,y_train)

        pred=model.predict(X_test)

        proba=model.predict_proba(X_test)

        scores=evaluate(

            y_test,

            pred,

            proba

        )

        results[name]=scores

        if scores["ROC AUC"]>best_auc:

            best_auc=scores["ROC AUC"]

            best_model=model

    dump(best_model,"models/best_model.pkl")

    return results