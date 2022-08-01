"""
This is a boilerplate pipeline 'TrainMyModels'
generated using Kedro 0.18.1
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import tune_lgbm_with_optuna, tune_logistic_regression_with_optuna, make_xtr_and_false_xval,\
    ensemble_validation, define_scalers_and_list_of_features_based_on_outliers, \
    make_kaggle_submission, logistic_regression_validation, lgbm_validation

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        
        node(
            func=make_xtr_and_false_xval,
            inputs="fe_train",
            outputs=["xtr", "false_xval", "ytr", "false_yval"],
            name="makeXtrAndFalseXval"
        ),

        node(
            func=define_scalers_and_list_of_features_based_on_outliers,
            inputs="xtr",
            outputs=["robust_scaler", "min_max_scaler", "robust_scaler_features_names", "min_max_scaler_features_names"],
            name="defineScalersBasedOnOutliers"
        ),

        node(
            func=tune_logistic_regression_with_optuna,
            inputs=["xtr", "ytr", "fe_train", "params:splits_for_optuna", "robust_scaler_features_names", "min_max_scaler_features_names", "correlation_relatory"],
            outputs="tuned_lr",
            name="TuneLrWithOptuna"
        ),

        node(
            func=tune_lgbm_with_optuna,
            inputs=["xtr", "ytr", "params:splits_for_optuna", "params:trials_for_optuna", "robust_scaler", "robust_scaler_features_names", "min_max_scaler", "min_max_scaler_features_names"],
            outputs="tuned_lgbm",
            name="TuneLgbmWithOptuna"
        ),

        node(
            func=lgbm_validation,
            inputs=["false_xval", "false_yval", "tuned_lgbm", "robust_scaler", "robust_scaler_features_names", "min_max_scaler", "min_max_scaler_features_names"],
            outputs="lgbm_predictions", 
            name="LgbmValidation"
        ),

        node(
            func=logistic_regression_validation,
            inputs=["false_xval", "false_yval", "tuned_lr", "params:splits_for_validation", "robust_scaler", "robust_scaler_features_names", "min_max_scaler", "min_max_scaler_features_names"],
            outputs="lr_predictions",
            name="LogisticRegressionValidation"
        ),

        node(
            func=ensemble_validation,
            inputs=["false_xval", "false_yval", "tuned_lgbm", "tuned_lr", "robust_scaler", "robust_scaler_features_names", "min_max_scaler", "min_max_scaler_features_names"],
            outputs="ensemble_metrics",
            name="EnsembleValidation"
        ),

        node(
            func=make_kaggle_submission,
            inputs=["tuned_lgbm", "xval"],
            outputs="submission",
            name="MakeKaggleSubmission"
        ),

    ])
