import optuna

def objective(trial):

    params = {

        "max_depth":

        trial.suggest_int(

            "max_depth",

            3,

            10

        ),

        "learning_rate":

        trial.suggest_float(

            "learning_rate",

            0.01,

            0.3

        ),

        "n_estimators":

        trial.suggest_int(

            "n_estimators",

            100,

            600

        )

    }
study = optuna.create_study(

    direction="maximize"

)

study.optimize(

    objective,

    n_trials=50

)