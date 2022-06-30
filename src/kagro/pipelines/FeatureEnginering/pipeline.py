"""
This is a boilerplate pipeline 'FeatureEnginering'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_cleaning_and_imputing, test_cleaning_and_imputing, make_my_features, use_scalers_based_on_outliers


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
            func=make_my_features,
            inputs=["cleaned_train", "params:target_col", "params:top_ratio"],
            outputs="fe_train",
            name="makeMyFeatures"
            ),

        node(
            func=use_scalers_based_on_outliers,
            inputs="fe_train",
            outputs=["xtr", "ytr"],
            name="UseScalersBasedOnOutliers"
            ),

    ])
