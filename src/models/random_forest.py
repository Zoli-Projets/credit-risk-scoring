from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:

    def __init__(self):

        self.model = RandomForestClassifier(

            n_estimators=300,

            max_depth=8,

            random_state=42

        )

    def train(self, X_train, y_train):

        self.model.fit(X_train,y_train)

    def predict(self,X):

        return self.model.predict(X)

    def predict_proba(self,X):

        return self.model.predict_proba(X)[:,1]