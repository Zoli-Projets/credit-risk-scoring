from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score

)

def evaluate(y_true,y_pred,y_score):

    return {

        "Accuracy":accuracy_score(y_true,y_pred),

        "Precision":precision_score(y_true,y_pred),

        "Recall":recall_score(y_true,y_pred),

        "F1":f1_score(y_true,y_pred),

        "ROC AUC":roc_auc_score(y_true,y_score)

    }