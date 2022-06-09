"""
This is a boilerplate pipeline 'Ensemble'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import predict_using_ensemble


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func=predict_using_ensemble,
            inputs=["test", ["model1", "model2", "model3"]],
            outputs=["yhat"],
            name="PredictUsingEnsemble"
        ),

    ])
