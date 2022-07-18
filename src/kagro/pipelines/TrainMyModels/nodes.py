"""
This is a boilerplate pipeline 'TrainMyModels'
generated using Kedro 0.18.1
"""
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
import pickle
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, f1_score, log_loss
import lightgbm as lgb
import mlflow


def define_scalers_and_list_of_features_based_on_outliers(fe_train: pd.DataFrame):
    # Take all the numerical columns
    numericalColumns = [col for col in fe_train.columns if fe_train[col].dtype == np.number]

    # Find wich cols have a good number of outliers
    robustScalerFeatures = []
    minMaxScalerFeatures = []

    for ncol in numericalColumns:
        # Calculate Q1 and Q3
        Q1 = fe_train[ncol].quantile(0.25)
        Q3 = fe_train[ncol].quantile(0.75)

        # Count the number of outliers
        if len(fe_train[ncol].loc[(fe_train[ncol] < Q1) | (fe_train[ncol] > Q3)]) > 2:
            robustScalerFeatures.append(ncol)
        else:
            minMaxScalerFeatures.append(ncol)
    
    # Substitue infinities with specific value
    fe_train.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Split train and test
    X = fe_train[[col for col in fe_train.columns if col != "target"]]
    del fe_train

    # Fit Robust scaler and save feature names as property
    r = RobustScaler()
    r.fit(X[robustScalerFeatures])

    # Fit MinMaxScaler scaler and save feature names as property
    m = MinMaxScaler()
    m.fit(X[minMaxScalerFeatures])

    return r, m, pd.DataFrame({"features": robustScalerFeatures}), pd.DataFrame({"features": minMaxScalerFeatures})


def use_scalers_at_train_and_test(fe_train: pd.DataFrame, fe_test: pd.DataFrame, 
                                robust_scaler: pickle, min_max_scaler: pickle, 
                                robust_scaler_features_names: pd.DataFrame, min_max_scaler_features_names: pd.DataFrame):
    # Substitue infinities with 0
    fe_train.replace([np.inf, -np.inf], 0, inplace=True)
    fe_test.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Define XTR and XVAL
    XTR = fe_train[[col for col in fe_train.columns if col != "target"]]
    YTR = pd.DataFrame(fe_train["target"])
    XVAL = fe_test
    del fe_train, fe_test

    # Use scalers at train and test
    XTR[robust_scaler_features_names["features"]] = robust_scaler.transform(XTR[robust_scaler_features_names["features"]])
    XTR[min_max_scaler_features_names["features"]] = min_max_scaler.transform(XTR[min_max_scaler_features_names["features"]])
    XVAL[robust_scaler_features_names["features"]] = robust_scaler.transform(XVAL[robust_scaler_features_names["features"]])
    XVAL[min_max_scaler_features_names["features"]] = min_max_scaler.transform(XVAL[min_max_scaler_features_names["features"]])

    return XTR, YTR, XVAL


def tune_lgbm_with_optuna(xtr: pd.DataFrame, ytr: pd.DataFrame, fe_train: pd.DataFrame, 
                        splits: int, robust_scaler_features_names: pd.DataFrame, 
                        min_max_scaler_features_names: pd.DataFrame) -> pickle:

    # Define optuna optimization function
    def optimize_function(trial, data=fe_train, splits=splits) -> float:
        # Instantiate grid of paremeters | Variable to append scores on
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

        # Delete inf values from train
        data.replace([np.inf, -np.inf], 0, inplace=True)

        # Separate xtr, and ytr
        XTR = data[[col for col in data.columns if col != "target"]]
        YTR = data["target"]

        # Instantiate Kfold
        kf = KFold(n_splits=splits)

        # Iterate making train and test partitions
        for idx, (tr_ind, val_ind) in enumerate (kf.split(XTR, YTR)):
            # Separate train and test
            xtr, xval = XTR.iloc[tr_ind], XTR.iloc[val_ind]
            ytr, yval = YTR.iloc[tr_ind], YTR.iloc[val_ind]

            # Fit robust and min_max_scaler at train
            r = RobustScaler()
            r.fit(xtr[robust_scaler_features_names["features"]])
            xtr[robust_scaler_features_names["features"]] = r.transform(xtr[robust_scaler_features_names["features"]])

            m = MinMaxScaler()
            m.fit(xtr[min_max_scaler_features_names["features"]])
            xtr[min_max_scaler_features_names["features"]] = m.transform(xtr[min_max_scaler_features_names["features"]])

            # Transform xval with scalers
            xval[robust_scaler_features_names["features"]] = r.transform(xval[robust_scaler_features_names["features"]])
            xval[min_max_scaler_features_names["features"]] = m.transform(xval[min_max_scaler_features_names["features"]])

            # Make light gbm with paremeters as paramGrid**
            lgbm = lgb.LGBMClassifier(objective="binary", **paramGrid)

            # fit
            lgbm.fit(xtr, ytr, eval_set=[(xval, yval)], eval_metric="binary_logloss", callbacks=[LightGBMPruningCallback(trial, "binary_logloss")])

            # Make yhat
            yhat = lgbm.predict_proba(xval)[:, 1]
            scores[idx] = log_loss(yval, yhat)

        # Return the mean of scores in all 10 validations
        return np.mean(scores)

    # Instantiate a optuna study
    study = optuna.create_study(direction="minimize", study_name="LGBM Optuna Bayesian Optimization")
    func = lambda trial: optimize_function(trial, data=fe_train, splits=splits)
    study.optimize(func, n_trials=30)

    # Save trained model with specified paremeters
    lgbm = lgb.LGBMClassifier(**study.best_params)
    final_model = lgbm.fit(xtr, ytr)

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


def kfold_10_mlflow_validation(fe_train: pd.DataFrame, lgbm: pickle, lr_regression: pickle, robust_scaler_features_names: pd.DataFrame, min_max_scaler_features_names: pd.DataFrame) -> pd.DataFrame:
    # def amex metric function
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
    
    # Substitue infinities with specific value
    fe_train.replace([np.inf, -np.inf], 0, inplace=True)

    # Instantiate kfold
    kf = KFold(n_splits=10, shuffle=True, random_state=32)
    sf = StratifiedKFold(n_splits=10, shuffle=True, random_state=32)

    # Iterate doing kfold | Stratified Kfold | Save metrics | And 
    metrics = {
        "split_type": [],
        "split_rep": [],
        "amex": [],
        "f1": [],
        "auc": []
    }

    # Define X and Y
    x = fe_train[[col for col in fe_train.columns if col != "target"]]
    y = fe_train["target"]


    # Do kfold validation
    kfCounter =  0
    # Instantiate matrix of zeros to use ensemble
    secondLevel = np.zeros((x.shape[0], 2))
    for tr_index, val_index in kf.split(x):
        # Start mlfflow nested run
        mlflow.start_run(run_name=f"Kfold_{kfCounter}", nested=True)

        # Separate train and test
        xtr, xval = x.iloc[tr_index], x.iloc[val_index]
        ytr, yval = y.iloc[tr_index], y.iloc[val_index]

        # Fit scalers at train | Use columns defined in the past node
        r = RobustScaler()
        r.fit(xtr[robust_scaler_features_names["features"]])

        m = MinMaxScaler()
        m.fit(xtr[min_max_scaler_features_names["features"]])

        # Apply scalers on validation
        xval[robust_scaler_features_names["features"]] = r.transform(xval[robust_scaler_features_names["features"]])
        xval[min_max_scaler_features_names["features"]] = m.transform(xval[min_max_scaler_features_names["features"]])

        # Make yhat of all models
        yhat_lgbm = lgbm.predict_proba(xval)[:, 1]
        yhat_lr = lr_regression.predict_proba(xval[lr_regression.features])[:, 1]

        # Save yhat at matrice
        secondLevel[val_index, 0] = yhat_lgbm
        secondLevel[val_index, 1] = yhat_lr

        # take xtr, xval, try, yval of matrice
        _s_xtr, _s_xval = secondLevel[tr_index], secondLevel[val_index]

        # Make secondLevelLogisticRegression | and fit it
        secondLevellr = LogisticRegression()
        secondLevellr.fit(_s_xtr, ytr)
        yhat_proba = secondLevellr.predict_proba(_s_xval)[:, 1]
        yhat = (yhat_proba > 0.5).astype(int)

        # Def predictions
        xval["yhat"] = yhat 
        xval["prediction"] = yhat_proba
        xval["target"] = yval

        # Eval metrics
        auc = roc_auc_score(yhat, yval)
        f1 = f1_score(yhat, yval)
        amex = amex_metric(pd.DataFrame(xval["target"]), pd.DataFrame(xval["prediction"]))

        # Save metrics
        metrics["split_type"].append("Kfold")
        metrics["split_rep"].append(kfCounter)
        metrics["amex"].append(amex)
        metrics["auc"].append(auc)
        metrics["f1"].append(f1)

        # Log metrics to mlflow
        mlflow.log_metric("amex", amex)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("f1", f1)

        # Evaluate Precision between categorys
        catCols = ['B_30', 'B_38',
        'D_114', 'D_116', 'D_117']
        # 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

        # Iterate each category col logging a metrics graph of them
        for catCol in catCols:
            uniqueCatsInCatCol = xval[catCol].unique()

            # Iterate over categorys calculating the metrics in them
            dictForGraph = {"category": [], "metric": [], "value": []}

            for uniqueCat in uniqueCatsInCatCol:
                # instantiate temp xval target and temp xval pred
                _temp_target = xval["target"].loc[xval[catCol] == uniqueCat]
                _temp_prev = xval["yhat"].loc[xval[catCol] == uniqueCat]

                # log auc
                dictForGraph["category"].append(uniqueCat)
                dictForGraph["metric"].append("auc")
                dictForGraph["value"].append(roc_auc_score(_temp_prev, _temp_target))

                # log f1
                dictForGraph["category"].append(uniqueCat)
                dictForGraph["metric"].append("f1")
                dictForGraph["value"].append(f1_score(_temp_prev, _temp_target))

                # log amex
                dictForGraph["category"].append(uniqueCat)
                dictForGraph["metric"].append("amex")
                dictForGraph["value"].append(amex_metric(
                                            pd.DataFrame(xval["target"].loc[xval[catCol] == uniqueCat]), 
                                            pd.DataFrame(xval["prediction"].loc[xval[catCol] == uniqueCat])
                                                )
                                            )
            
                # del
                del _temp_target, _temp_prev

            # Generate seaborn graph and log it into mlflow
            dictForGraph = pd.DataFrame(dictForGraph)
            fig = px.bar(dictForGraph, x=dictForGraph["category"], y=dictForGraph["value"], barmode="group", color="metric")
            mlflow.log_figure(fig, f"auc_per_category_{catCol}.html")

        mlflow.end_run()
        kfCounter += 1


    # Do Stratified kfold validation!
    sfCounter =  0
    # Instantiate matrix of zeros to use ensemble
    secondLevel = np.zeros((x.shape[0], 2))
    for tr_index, val_index in sf.split(x, y):
        # Start mlfflow nested run
        mlflow.start_run(run_name=f"Stratified_Kfold_{sfCounter}", nested=True)

        # Separate train and test
        xtr, xval = x.iloc[tr_index], x.iloc[val_index]
        ytr, yval = y.iloc[tr_index], y.iloc[val_index]

        # Fit scalers at train | Use columns defined in the past node
        r = RobustScaler()
        r.fit(xtr[robust_scaler_features_names["features"]])

        m = MinMaxScaler()
        m.fit(xtr[min_max_scaler_features_names["features"]])

        # Apply scalers on validation
        xval[robust_scaler_features_names["features"]] = r.transform(xval[robust_scaler_features_names["features"]])
        xval[min_max_scaler_features_names["features"]] = m.transform(xval[min_max_scaler_features_names["features"]])

        # Make yhat of all models
        yhat_lgbm = lgbm.predict_proba(xval)[:, 1]
        yhat_lr = lr_regression.predict_proba(xval[lr_regression.features])[:, 1]

        # Save yhat at matrice
        secondLevel[val_index, 0] = yhat_lgbm
        secondLevel[val_index, 1] = yhat_lr

        # take xtr, xval, try, yval of matrice
        _s_xtr, _s_xval = secondLevel[tr_index], secondLevel[val_index]

        # Make secondLevelLogisticRegression | and fit it
        secondLevellr = LogisticRegression()
        secondLevellr.fit(_s_xtr, _s_xval)
        yhat_proba = secondLevellr.predict_proba(_s_xval)[:, 1]
        yhat = (yhat_proba > 0.5).astype(int)

        # Def predictions
        xval["yhat"] = yhat 
        xval["prediction"] = yhat_proba
        xval["target"] = yval

        # Eval metrics
        auc = roc_auc_score(yhat, yval)
        f1 = f1_score(yhat, yval)
        amex = amex_metric(pd.DataFrame(xval["target"]), pd.DataFrame(xval["prediction"]))

        # Save metrics
        metrics["split_type"].append("Stratified_Kfold")
        metrics["split_rep"].append(sfCounter)
        metrics["amex"].append(amex)
        metrics["auc"].append(auc)
        metrics["f1"].append(f1)

        # Log metrics to mlflow
        mlflow.log_metric("amex", amex)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("f1", f1)

        # Evaluate Precision between categorys
        catCols = ['B_30', 'B_38',
        'D_114', 'D_116', 'D_117']
        # 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']

        # Iterate each category col logging a metrics graph of them
        for catCol in catCols:
            uniqueCatsInCatCol = xval[catCol].unique()

            # Iterate over categorys calculating the metrics in them
            dictForGraph = {"category": [], "metric": [], "value": []}

            for uniqueCat in uniqueCatsInCatCol:
                # instantiate temp xval target and temp xval pred
                _temp_target = xval["target"].loc[xval[catCol] == uniqueCat]
                _temp_prev = xval["yhat"].loc[xval[catCol] == uniqueCat]

                # log auc
                dictForGraph["category"].append(uniqueCat)
                dictForGraph["metric"].append("auc")
                dictForGraph["value"].append(roc_auc_score(_temp_prev, _temp_target))

                # log f1
                dictForGraph["category"].append(uniqueCat)
                dictForGraph["metric"].append("f1")
                dictForGraph["value"].append(f1_score(_temp_prev, _temp_target))

                # log amex
                dictForGraph["category"].append(uniqueCat)
                dictForGraph["metric"].append("amex")
                dictForGraph["value"].append(amex_metric(
                                            pd.DataFrame(xval["target"].loc[xval[catCol] == uniqueCat]), 
                                            pd.DataFrame(xval["prediction"].loc[xval[catCol] == uniqueCat])
                                                )
                                            )
            
                # del
                del _temp_target, _temp_prev

            # Generate seaborn graph and log it into mlflow
            dictForGraph = pd.DataFrame(dictForGraph)
            fig = px.bar(dictForGraph, x=dictForGraph["category"], y=dictForGraph["value"], barmode="group", color="metric")
            mlflow.log_figure(fig, f"auc_per_category_{catCol}.html")

        mlflow.end_run()
        sfCounter += 1

    # Transform metrics into datframe
    metrics = pd.DataFrame(metrics)

    # Log mlflow mean of metrics
    for m in ["amex", "f1", "auc"]:
        mlflow.log_metric(m, metrics[m].mean())

    return metrics


def make_kaggle_submission(model: pickle, xval: pd.DataFrame) -> pd.DataFrame:
    # Make submission
    yhat = model.predict_proba(xval[[col for col in xval.columns if "ID" not in col]])[:, 1]

    # Make sample submission
    xval["prediction"] = yhat
    submission = xval[["customer_ID", "prediction"]]

    return submission