"""
This is a boilerplate pipeline 'TrainMyModels'
generated using Kedro 0.18.1
"""
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score
import lightgbm as lgb
import mlflow

# Amex metric
def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)


def fit_raw_light_gbm(data: pd.DataFrame):
    """
    Take a train input and returns a trained pickled model
    """
    pass

def kfold_10_validation(x: pd.DataFrame, y: pd.DataFrame):
    # Make model and log it into mlflow
    model = lgb.LGBMClassifier(num_leaves=98, max_depth=- 1, learning_rate=2, n_estimators=10)
    mlflow.lightgbm.autolog(model)

    # Instantiate kfold
    kf = KFold(n_splits=10)

    # Iterate doing kfold | Save metrics
    metrics = {
        "amex": [],
        "f1": [],
        "auc": []
    }
    
    for tr_index, val_index in kf.split(x):
        # Separate train and test
        xtr, xval = x.iloc[tr_index], x.iloc[val_index]
        ytr, yval = y.iloc[tr_index], y.iloc[val_index]

        # Fit the model
        model.fit(xtr, ytr)

        # Yhat | column to evaluate prediction
        yhat_proba = model.predict_proba(xval)[:, 1]
        yhat = model.predict(xval)
        xval["prediction"] = yhat_proba
        xval["target"] = yhat_proba

        # Eval metrics
        auc = roc_auc_score(yhat, yval)
        f1 = f1_score(yhat, yval)
        amex = amex_metric(pd.DataFrame(xval["target"]), pd.DataFrame(xval["prediction"]))

        # Save metrics
        metrics["amex"].append(amex)
        metrics["auc"].append(auc)
        metrics["f1"].append(f1)

    # Log metrics into mlflow
    mlflow.log_metric("amex", sum(metrics["amex"]) / len(metrics["amex"]))
    mlflow.log_metric("f1", sum(metrics["f1"]) / len(metrics["f1"]))
    mlflow.log_metric("auc", sum(metrics["auc"]) / len(metrics["auc"]))


        
    return None
