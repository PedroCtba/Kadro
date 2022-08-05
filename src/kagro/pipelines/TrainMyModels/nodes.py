"""
This is a boilerplate pipeline 'TrainMyModels'
generated using Kedro 0.18.1
"""
import pandas as pd
import numpy as np
from random import randint
import pickle
import optuna
import lightgbm as lgb
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, f1_score, log_loss

import tensorflow as tf
from ..global_functions import amex_metric


def make_xtr_and_false_xval(fe_train: pd.DataFrame):
    # Use 66% of known data for train, and 33% for test (This test data is the "false" xtrain, as we known the true result, but dont want to use it as part of tunning, scaling, and etc)
    X = fe_train[[col for col in fe_train.columns if "target" not in col]]
    Y = fe_train["target"]
    xtr, false_xval, ytr, false_yval = train_test_split(X, Y, test_size=0.33, random_state=32)

    # Return train data, and "false" test data
    return xtr, false_xval, ytr, false_yval


def define_scalers_and_list_of_features_based_on_outliers(xtr: pd.DataFrame):
    # Take all the numerical columns from train
    numericalColumns = [col for col in xtr.columns if xtr[col].dtype == np.number]

    # Find wich cols have a good number of outliers
    robustScalerFeatures = []
    minMaxScalerFeatures = []

    for ncol in numericalColumns:
        # Calculate Q1 and Q3
        Q1 = xtr[ncol].quantile(0.25)
        Q3 = xtr[ncol].quantile(0.75)

        # Count the number of outliers
        if len(xtr[ncol].loc[(xtr[ncol] < Q1) | (xtr[ncol] > Q3)]) > 2:
            robustScalerFeatures.append(ncol)

        else:
            minMaxScalerFeatures.append(ncol)
    
    # Split train and test
    X = xtr[[col for col in xtr.columns if col != "target"]]
    del xtr

    # Fit Robust scaler
    r = RobustScaler()
    r.fit(X[robustScalerFeatures])

    # Fit MinMaxScaler scaler
    m = MinMaxScaler()
    m.fit(X[minMaxScalerFeatures])

    return r, m, pd.DataFrame({"features": robustScalerFeatures}), pd.DataFrame({"features": minMaxScalerFeatures})


def tune_lgbm_with_optuna(xtr: pd.DataFrame, ytr: pd.DataFrame, splits: int, trials: int, 
    robust_scaler: pickle, robust_scaler_features_names: pd.DataFrame, 
    min_max_scaler: pickle, min_max_scaler_features_names: pd.DataFrame) -> pickle:
    
    # Define optuna optimization function
    def optimize_function(trial, x=xtr, y=ytr, splits=splits) -> float:
        # Instantiate grid of parameters | Variable to append scores on
        paramGrid = {
            "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
            "max_bin": trial.suggest_int("max_bin", 200, 300),
            "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
            "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.2, 0.95, step=0.1
            ),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.2, 0.95, step=0.1
            ),
        }
        scores = np.empty(splits)

        # Instantiate Kfold
        kf = KFold(n_splits=splits)

        # Iterate making train and test partitions
        for idx, (tr_ind, val_ind) in enumerate (kf.split(x, y)):
            # Separate train and test data
            xtr, xval = x.iloc[tr_ind], x.iloc[val_ind]
            ytr, yval = y.iloc[tr_ind], y.iloc[val_ind]

            # Fit robust and min_max_scaler at train
            r = RobustScaler()
            r.fit(xtr[robust_scaler_features_names["features"]])
            xtr[robust_scaler_features_names["features"]] = r.transform(xtr[robust_scaler_features_names["features"]])

            m = MinMaxScaler()
            m.fit(xtr[min_max_scaler_features_names["features"]])
            xtr[min_max_scaler_features_names["features"]] = m.transform(xtr[min_max_scaler_features_names["features"]])

            # Transform xval with fited sacalars
            xval[robust_scaler_features_names["features"]] = r.transform(xval[robust_scaler_features_names["features"]])
            xval[min_max_scaler_features_names["features"]] = m.transform(xval[min_max_scaler_features_names["features"]])

            # Make lgbm with paremeters as paramGrid**
            lgbm = lgb.LGBMClassifier(objective="binary", **paramGrid)

            # fit
            lgbm.fit(xtr, ytr, eval_set=[(xval, yval)], eval_metric="binary_logloss", callbacks=[optuna.integration.LightGBMPruningCallback(trial, "binary_logloss")])

            # Make yhat
            yhat = lgbm.predict_proba(xval)[:, 1]
            scores[idx] = log_loss(yval, yhat)

        # Return the mean of scores on all kfold parts
        return np.mean(scores)

    # Instantiate a optuna study
    study = optuna.create_study(direction="minimize", study_name="LGBM Optuna Bayesian Optimization")
    func = lambda trial: optimize_function(trial, x=xtr, y=ytr, splits=splits)
    study.optimize(func, n_trials=trials)

    # Instantiate a model with optuna parameters
    lgbm = lgb.LGBMClassifier(**study.best_params)

    # Use scalers at x train
    xtr[robust_scaler_features_names["features"]] = robust_scaler.transform(xtr[robust_scaler_features_names["features"]])
    xtr[min_max_scaler_features_names["features"]] = min_max_scaler.transform(xtr[min_max_scaler_features_names["features"]])

    # Fit lgbm
    final_model = lgbm.fit(xtr, ytr)

    # Return tuned lgbm
    return final_model


def tune_logistic_regression_with_optuna(xtr: pd.DataFrame, ytr: pd.DataFrame, 
    fe_train:pd.DataFrame, splits: int, robust_scaler_features_names: pd.DataFrame, 
    min_max_scaler_features_names: pd.DataFrame, correlationRelatory: pd.DataFrame):

    # Select list of variables from correlation dataframe, in asceding order by correaltion force, drop variables that di not rejected null H0 in kruskal
    correlationRelatory = correlationRelatory.loc[correlationRelatory["kruskal_reject_h0"] == 1]
    correlationRelatory = correlationRelatory.sort_values("BiCorr", ascending=False)
    selectedVariables = correlationRelatory["Column"].tolist()
    
    # Iterate selecting variables with positive correalation, and dropping variables with high corr with already selected variables
    for variable in selectedVariables:
        for testVariable in selectedVariables:
            if xtr[variable].corr(xtr[testVariable]) > 0.30:
                selectedVariables.remove(testVariable)

    # Define optuna Function to tune losgistic regression
    def optimize_function(trial, data=fe_train, splits=splits, selected_cols=selectedVariables) -> float:
        # Instantiate grid of paremeters
        paramGrid = {
            "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
            "penalty": trial.suggest_categorical("penalty", ["none", "l1", "l2", "elasticnet"]),
            "C": trial.suggest_float("C", 0.001, 10),
        }
        scores = np.empty(splits)

        # Delete inf values from train
        data.replace([np.inf, -np.inf], 0, inplace=True)

        # Separate xtr, and ytr
        XTR = data[[col for col in data.columns if col != "target" and col in selected_cols]]
        YTR = data["target"]

        # Instantiate Kfold
        kf = KFold(n_splits=splits)

        # Iterate making train and test partitions
        for idx, (tr_ind, val_ind) in enumerate(kf.split(XTR, YTR)):
            # Separate train and test
            xtr, xval = XTR.iloc[tr_ind], XTR.iloc[val_ind]
            ytr, yval = YTR.iloc[tr_ind], YTR.iloc[val_ind]

            # Fit robust and min_max_scaler at train
            r = RobustScaler()
            rFeatures = [f for f in robust_scaler_features_names["features"] if f in selected_cols]
            r.fit(xtr[rFeatures])
            xtr[rFeatures] = r.transform(xtr[rFeatures])

            m = MinMaxScaler()
            mFeatures = [f for f in min_max_scaler_features_names["features"] if f in selected_cols]
            m.fit(xtr[mFeatures])
            xtr[mFeatures] = m.transform(xtr[mFeatures])

            # Transform xval with scalers
            xval[rFeatures] = r.transform(xval[rFeatures])
            xval[mFeatures] = m.transform(xval[mFeatures])

            # Make logistic regression with paremeters as paramGrid**
            lr = LogisticRegression(**paramGrid)

            # fit
            lr.fit(xtr, ytr)

            # Make yhat
            yhat = lr.predict_proba(xval)[:, 1]
            scores[idx] = log_loss(yval, yhat)

        # Return the mean of scores in all 10 validations
        return np.mean(scores)

    # Instantiate a optuna study
    study = optuna.create_study(direction="minimize", study_name="Logistic Regression Optuna Bayesian Optimization")
    func = lambda trial: optimize_function(trial, data=fe_train, splits=splits, selected_cols=selectedVariables)
    study.optimize(func, n_trials=30)

    # Save trained model with specified paremeters
    lr = LogisticRegression(
        # **study.best_params
        )
    final_model = lr.fit(xtr[selectedVariables], ytr)
    final_model.features = selectedVariables

    return final_model


def train_neural_network(xtr: pd.DataFrame, ytr: pd.DataFrame,
    robust_scaler: pickle, robust_scaler_features_names: pd.DataFrame, 
    min_max_scaler: pickle, min_max_scaler_features_names: pd.DataFrame) -> pickle:
    # Def funcitons to return model
    def make_neural_network():
        # Input layer
        inp = tf.keras.layers.Input((xtr.shape[1], ))

        # Make hidden layer
        hid = tf.keras.layers.Dense(int(xtr.shape[1]/2), activation="relu")(inp)

        # Make output layer
        out = tf.keras.layers.Dense(1, activation="sigmoid")(hid)

        # Make model
        nn = tf.keras.Model(inp, out)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = tf.keras.losses.BinaryCrossentropy()
        nn.compile(loss=loss, optimizer = opt)

        return nn
    
    # Use mdethod to make model
    nn = make_neural_network()
    
    # Make early stopping mechanism
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, mode="min", restore_best_weights=True)

    # Scale the training data
    xtr[robust_scaler_features_names["features"]] = robust_scaler.transform(xtr[robust_scaler_features_names["features"]])
    xtr[min_max_scaler_features_names["features"]] = min_max_scaler.transform(xtr[min_max_scaler_features_names["features"]])

    # Fit neural network
    nn = nn.fit(xtr, ytr, epochs=200, shuffle=True, batch_size=1, callbacks=[es])


def logistic_regression_validation(false_xval: pd.DataFrame, false_yval:pd.DataFrame, 
    lr_regression: pickle, splits_for_validation: int,
    robust_scaler: pickle, robust_scaler_features_names: pd.DataFrame,
    min_max_scaler: pickle, min_max_scaler_features_names: pd.DataFrame) -> pd.DataFrame:

    # Define sets of index for xval testing
    index_lists = []
    val_size = int(len(false_yval) / splits_for_validation)
    for split in range(splits_for_validation):
        # Make current index list | append it to index_lists
        _index = [randint(0, len(false_yval) -1) for _ in range(val_size)]
        index_lists.append(_index)

    # Iterate all the validation sets
    lr_predictions = {"index": [], "predictions": []}

    # Instantiate metrics
    tot_amex = 0
    tot_auc = 0
    tot_f1 = 0

    for val_rep, val_set in enumerate(index_lists):
        # Start mlflow nested run
        mlflow.start_run(run_name=f"Log Regresison {str(val_rep)} validation", nested=True)

        # Separate train and test data
        _xval, _yval = false_xval.iloc[val_set], false_yval.iloc[val_set]

        # Transform false xval with scalers | Dont fit since the model is not prepared for a not know distribution
        _xval[robust_scaler_features_names["features"]] = robust_scaler.transform(_xval[robust_scaler_features_names["features"]])
        _xval[min_max_scaler_features_names["features"]] = min_max_scaler.transform(_xval[min_max_scaler_features_names["features"]])

        # Predict using trained logistic regression
        yhat_proba = lr_regression.predict_proba(_xval[lr_regression.features])[:, 1]
        yhat_not_proba = (yhat_proba > 0.5).astype(int)

        # Add predicitons to validaiton dataframe (to evaluate amex metric)
        _xval["yhat"] = yhat_not_proba
        _xval["prediction"] = yhat_proba
        _xval["target"] = _yval

        # Eval metrics
        auc = roc_auc_score(_yval, yhat_proba)
        f1 = f1_score(_yval, yhat_not_proba)
        amex = amex_metric(pd.DataFrame(_xval["target"]), pd.DataFrame(_xval["prediction"]))

        # Log metrics to mlflow
        mlflow.log_metric("amex", amex)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("f1", f1)

        # Sum metrics at variables, log meand latter to mlflow
        tot_amex += amex
        tot_auc += auc
        tot_f1 += f1

        # Save model predictions
        lr_predictions["index"] += [i for i in _xval.index]
        lr_predictions["predictions"] += [i for i in yhat_proba]

        # End mlflow run
        mlflow.end_run()

    # Log mean of all metrics at mlflow
    mlflow.log_metric("amex", tot_amex/(val_rep+1))
    mlflow.log_metric("auc", tot_auc/(val_rep+1))
    mlflow.log_metric("f1", tot_f1/(val_rep+1))

    # Convert model predictions to dataframe
    lr_predictions = pd.DataFrame(lr_predictions)

    # Return model predictions (See corelations with other predictions latter)
    return lr_predictions


def lgbm_validation(false_xval: pd.DataFrame, false_yval:pd.DataFrame, 
    lgbm: pickle, splits_for_validation: int,
    robust_scaler: pickle, robust_scaler_features_names: pd.DataFrame,
    min_max_scaler: pickle, min_max_scaler_features_names: pd.DataFrame) -> pd.DataFrame:

    # Define sets of index for xval testing
    index_lists = []
    val_size = int(len(false_yval) / splits_for_validation)
    for split in range(splits_for_validation):
        # Make current index list | append it to index_lists
        _index = [randint(0, len(false_yval) -1) for _ in range(val_size)]
        index_lists.append(_index)

    # Iterate all the validation sets
    lgbm_predictions = {"index": [], "predictions": []}

    # Instantiate metrics
    tot_amex = 0
    tot_auc = 0
    tot_f1 = 0

    for val_rep, val_set in enumerate(index_lists):
        # Start mlflow nested run
        mlflow.start_run(run_name=f"LGBM {str(val_rep)} validation", nested=True)

        # Separate train and test data
        _xval, _yval = false_xval.iloc[val_set], false_yval.iloc[val_set]

        # Transform false xval with scalers | Dont fit since the model is not prepared for a not know distribution
        _xval[robust_scaler_features_names["features"]] = robust_scaler.transform(_xval[robust_scaler_features_names["features"]])
        _xval[min_max_scaler_features_names["features"]] = min_max_scaler.transform(_xval[min_max_scaler_features_names["features"]])

        # Predict using trained lgbm
        yhat_proba = lgbm.predict_proba(_xval)[:, 1]
        yhat_not_proba = (yhat_proba > 0.5).astype(int)

        # Add predicitons to validaiton dataframe (to evaluate amex metric)
        _xval["yhat"] = yhat_not_proba
        _xval["prediction"] = yhat_proba
        _xval["target"] = _yval

        # Eval metrics
        auc = roc_auc_score(_yval, yhat_proba)
        f1 = f1_score(_yval, yhat_not_proba)
        amex = amex_metric(pd.DataFrame(_xval["target"]), pd.DataFrame(_xval["prediction"]))

        # Log metrics to mlflow
        mlflow.log_metric("amex", amex)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("f1", f1)

        # Sum metrics at variables, log meand latter to mlflow
        tot_amex += amex
        tot_auc += auc
        tot_f1 += f1

        # Save model predictions
        lgbm_predictions["index"] += [i for i in _xval.index]
        lgbm_predictions["predictions"] += [i for i in yhat_proba]

        # End mlflow run
        mlflow.end_run()

    # Log mean of all metrics at mlflow
    mlflow.log_metric("amex", tot_amex/(val_rep+1))
    mlflow.log_metric("auc", tot_auc/(val_rep+1))
    mlflow.log_metric("f1", tot_f1/(val_rep+1))

    # Convert model predictions to dataframe
    lgbm_predictions = pd.DataFrame(lgbm_predictions)

    # Return model predictions (See corelations with other predictions latter)
    return lgbm_predictions


def ensemble_validation(false_xval: pd.DataFrame, false_yval:pd.DataFrame, 
    lgbm: pickle, lr_regression: pickle,
    robust_scaler: pickle, robust_scaler_features_names: pd.DataFrame, 
    min_max_scaler: pickle, min_max_scaler_features_names: pd.DataFrame) -> pd.DataFrame:

    # Instantiate matrix of zeros to use ensemble
    secondLevel = np.zeros((false_xval.shape[0], 2))

    # Transform false xval with scalers | Dont fit since the model is not prepared for a not know distribution
    false_xval[robust_scaler_features_names["features"]] = robust_scaler.transform(false_xval[robust_scaler_features_names["features"]])
    false_xval[min_max_scaler_features_names["features"]] = min_max_scaler.transform(false_xval[min_max_scaler_features_names["features"]])

    # Predict using trained lgbm | logisitc regression | neural network
    yhat_lgbm = lgbm.predict_proba(false_xval)[:, 1]
    yhat_lr = lr_regression.predict_proba(false_xval[lr_regression.features])[:, 1]

    # Save yhat at matrice
    secondLevel[false_xval.index, 0] = yhat_lgbm
    secondLevel[false_xval.index, 1] = yhat_lr


def make_kaggle_submission(model: pickle, xval: pd.DataFrame) -> pd.DataFrame:
    # Make submission
    yhat = model.predict_proba(xval[[col for col in xval.columns if "ID" not in col]])[:, 1]

    # Make sample submission
    xval["prediction"] = yhat
    submission = xval[["customer_ID", "prediction"]]

    return submission