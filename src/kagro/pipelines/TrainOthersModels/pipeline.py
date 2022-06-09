"""
This is a boilerplate pipeline 'TrainOthersModels'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_others_nn, train_others_rf, train_others_xgb, train_others_lgbm

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func=train_others_nn,
            inputs="train",
            outputs=["trained_others_nn"],
            name="trainOthersNn"
        ),

        node(
            func=train_others_rf,
            inputs="train",
            outputs=["trained_others_rf"],
            name="trainOthersRf"
        ),

        node(
            func=train_others_xgb,
            inputs="train",
            outputs=["trained_others_xgb"],
            name="trainOthersXgb"
        ),

        node(
            func=train_others_lgbm,
            inputs="train",
            outputs=["trained_others_lgbm"],
            name="trainOthersLgbm"
        ),

    ])
