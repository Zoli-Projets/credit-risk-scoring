from imblearn.pipeline import Pipeline

from imblearn.combine import SMOTETomek

from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

def create_pipeline():

    pipeline = Pipeline(

        [

            ("scaler", StandardScaler()),

            ("sampling", SMOTETomek()),

            ("model", XGBClassifier())

        ]

    )

    return pipeline