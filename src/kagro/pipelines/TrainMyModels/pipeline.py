"""
This is a boilerplate pipeline 'TrainMyModels'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import kfold_10_mlflow_validation, define_scalers_based_on_outliers, use_scalers_at_train_and_test


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func=define_scalers_based_on_outliers,
            inputs="fe_train",
            outputs=["robust_scaler", "min_max_scaler"],
            name="defineScalersBasedOnOutliers"
            ),

        node(
            func=use_scalers_at_train_and_test,
            inputs=["fe_train", "fe_test", "robust_scaler", "min_max_scaler"],
            outputs=["xtr", "ytr", "xval"],
            name="useScalersAtTrainAndTest"
            ),

        node(
            func=kfold_10_mlflow_validation,
            inputs="fe_train",
            outputs=None,
            name="Kfold10MlflowValidation"
        ),


    ])
