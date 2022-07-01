"""
This is a boilerplate pipeline 'FeatureEnginering'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_cleaning_and_imputing, test_cleaning_and_imputing, define_fe_process_with_train, make_my_features_at_train, make_my_features_at_test


def create_pipeline(**kwargs) -> Pipeline:

    return pipeline([
        node(
            func=train_cleaning_and_imputing,
            inputs=["train", "train_labels"],
            outputs="cleaned_train",
            name="trainCleaningAndImputing"
            ),

        node(
            func=test_cleaning_and_imputing,
            inputs="test",
            outputs="cleaned_test",
            name="testCleaningAndImputing"
            ),

        node(
            func=define_fe_process_with_train,
            inputs=["cleaned_train", "params:target_col"],
            outputs="correlation_relatory",
            name="defineFeProcessWithTrain"
            ),

        node(
            func=make_my_features_at_train,
            inputs=["cleaned_train", "correlation_relatory", "params:top_ratio"],
            outputs="fe_train",
            name="makeMyFeaturesAtTrain"
            ),

        node(
            func=make_my_features_at_test,
            inputs=["cleaned_test", "correlation_relatory", "params:top_ratio"],
            outputs="fe_test",
            name="makeMyFeaturesAtTest"
            ),

    ])
