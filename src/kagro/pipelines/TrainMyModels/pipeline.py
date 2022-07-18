"""
This is a boilerplate pipeline 'TrainMyModels'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import tune_lgbm_with_optuna, tune_logistic_regression_with_optuna, kfold_10_mlflow_validation, define_scalers_and_list_of_features_based_on_outliers, use_scalers_at_train_and_test, make_kaggle_submission


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func=define_scalers_and_list_of_features_based_on_outliers,
            inputs="fe_train",
            outputs=["robust_scaler", "min_max_scaler", "robust_scaler_features_names", "min_max_scaler_features_names"],
            name="defineScalersBasedOnOutliers"
            ),

        node(
            func=use_scalers_at_train_and_test,
            inputs=["fe_train", "fe_test", "robust_scaler", "min_max_scaler", "robust_scaler_features_names", "min_max_scaler_features_names"],
            outputs=["xtr", "ytr", "xval"],
            name="useScalersAtTrainAndTest"
            ),

        node(
            func=tune_logistic_regression_with_optuna,
            inputs=["xtr", "ytr", "fe_train", "params:splits_for_optuna", "robust_scaler_features_names", "min_max_scaler_features_names", "correlation_relatory"],
            outputs="tuned_lr",
            name="TuneLrWithOptuna"
        ),

        node(
            func=tune_lgbm_with_optuna,
            inputs=["xtr", "ytr", "fe_train", "params:splits_for_optuna", "robust_scaler_features_names", "min_max_scaler_features_names"],
            outputs="tuned_lgbm",
            name="TuneLgbmWithOptuna"
        ),

        node(
            func=kfold_10_mlflow_validation,
            inputs=["fe_train", "tuned_lgbm", "tuned_lr", "robust_scaler_features_names", "min_max_scaler_features_names"],
            outputs="kfold_metrics",
            name="Kfold10MlflowValidation"
        ),

        node(
            func=make_kaggle_submission,
            inputs=["tuned_lgbm", "xval"],
            outputs="submission",
            name="MakeKaggleSubmission"
        ),

    ])
