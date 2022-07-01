"""
This is a boilerplate pipeline 'TrainMyModels'
generated using Kedro 0.18.1
"""
import pandas as pd
import pickle

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, f1_score
import lightgbm as lgb
import mlflow


def define_scalers_based_on_outliers(fe_train: pd.DataFrame) -> tuple(pickle, pickle):
    # Take all the numerical columns
    numericalColumns = [col for col in fe_train.columns if fe_train[col].dtype == np.number]

    # Find wich cols have a good number of outliers
    robustScaler = []
    minMaxScaler = []

    for ncol in numericalColumns:
        # Calculate Q1 and Q3
        Q1 = fe_train[ncol].quantile(0.25)
        Q3 = fe_train[ncol].quantile(0.75)

        # Count the number of outliers
        if len(fe_train[ncol].loc[(fe_train[ncol] < Q1) | (fe_train[ncol] > Q3)]) > 2:
            robustScaler.append(ncol)
        else:
            minMaxScaler.append(ncol)
    
    # Substitue infinities with specific value
    fe_train.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Split train and test
    X = fe_train[[col for col in fe_train.columns if col != "target"]]
    del fe_train

    # Fit Robust scaler and save feature names as property
    r = RobustScaler()
    r.fit(X[robustScaler])
    r.feature_names = robustScaler

    # Fit MinMaxScaler scaler and save feature names as property
    m = MinMaxScaler()
    m.fit(X[minMaxScaler])
    m.feature_names = minMaxScaler

    return r, m


def use_scalers_at_train_and_test(fe_train: pd.DataFrame, fe_test: pd.DataFrame, robust_scaler: pickle, min_max_scaler: pickle) -> tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame):
    # Substitue infinities with 0
    fe_train.replace([np.inf, -np.inf], 0, inplace=True)
    fe_test.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Define XTR and XVAL
    XTR = fe_train[[col for col in fe_train.columns if col != "target"]]
    YTR = pd.DataFrame(fe_train["target"])
    XVAL = fe_test
    del fe_train, fe_test

    # Use scalers at train and test
    XTR[robust_scaler.feature_names] = robust_scaler.transform(XTR[robust_scaler.feature_names])
    XTR[min_max_scaler.feature_names] = min_max_scaler.transform(XTR[min_max_scaler.feature_names])
    XVAL[robust_scaler.feature_names] = robust_scaler.transform(XVAL[robust_scaler.feature_names])
    XVAL[min_max_scaler.feature_names] = min_max_scaler.transform(XVAL[min_max_scaler.feature_names])

    return XTR, YTR, XVAL


def tune_model_with_optuna() -> pickle: # Use X and Y off train to tune a model based infull train
    pass


def kfold_10_mlflow_validation(fe_train: pd.DataFrame, model: pickle):
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

    #Log model into mlflow
    mlflow.lightgbm.autolog(model)

    # Instantiate kfold
    kf = KFold(n_splits=10)

    # Iterate doing kfold | Save metrics
    metrics = {
        "amex": [],
        "f1": [],
        "auc": []
    }
    # Define X and Y
    x = fe_train[[col for col in fe_train.columns if col != "target"]]
    y = fe_train["target"]

    for tr_index, val_index in kf.split(x):
        # Separate train and test
        xtr, xval = x.iloc[tr_index], x.iloc[val_index]
        ytr, yval = y.iloc[tr_index], y.iloc[val_index]

        # Fit and apply scalers

        
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
