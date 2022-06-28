"""
This is a boilerplate pipeline 'TrainMyModels'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import kfold_10_validation


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func=kfold_10_validation,
            inputs=["xtr", "ytr"],
            outputs=None,
            name="10foldWithMlFlow"
        ),
    ])
