from xgboost import XGBClassifier

class XGBoostModel:

    def __init__(self):

        self.model = XGBClassifier(

            random_state=42,

            eval_metric="logloss"

        )

    def train(self,X_train,y_train):

        self.model.fit(X_train,y_train)

    def predict(self,X):

        return self.model.predict(X)

    def predict_proba(self,X):

        return self.model.predict_proba(X)[:,1]