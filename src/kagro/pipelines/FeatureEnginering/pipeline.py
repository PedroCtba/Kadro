"""
This is a boilerplate pipeline 'FeatureEnginering'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import join_train_with_labels, make_my_features, make_others_features, join_all_features


def create_pipeline(**kwargs) -> Pipeline:

    return pipeline([
        node(
            func=join_train_with_labels,
            inputs=["train_features", "train_labels"],
            outputs="train_joined",
            name="joinTrainWithLabels"
            ),

        node(
            func=make_my_features,
            inputs=["train_joined", "params:target_col", "params:top_ratio"],
            outputs="train_my_features",
            name="makeMyFeatures"
            ),

        node(
            func=make_others_features,
            inputs="train_joined",
            outputs=["xtr_others_features", "xval_others_features"],
            name="makeOthersFeatures"
            ),

        node(
            func=join_all_features,
            inputs=["xtr_my_features", "xval_my_features", "xtr_others_features", "xval_others_features"],
            outputs=["train", "test"],
            name="joinAllFeatures"
        ),

    ])
