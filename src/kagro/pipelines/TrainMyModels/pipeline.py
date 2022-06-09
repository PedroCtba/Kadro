"""
This is a boilerplate pipeline 'TrainMyModels'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_nn, train_rf, train_xgb, train_lgbm


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func=train_nn,
            inputs="train",
            outputs=["trained_nn"],
            name="trainNn"
            ),

        node(
            func=train_rf,
            inputs="train",
            outputs=["trained_rf"],
            name="trainRf"
        ),

        node(
            func=train_xgb,
            inputs="train",
            outputs=["trained_xgb"],
            name="trainXgb"
        ),

        node(
            func=train_lgbm,
            inputs="train",
            outputs=["trained_lgbm"],
            name="trainLgbm"
        ),

    ])
