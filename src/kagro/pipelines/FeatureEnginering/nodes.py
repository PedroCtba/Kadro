"""
This is a boilerplate pipeline 'FeatureEnginering'
generated using Kedro 0.18.1
"""
import pandas as pd


def make_my_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return the competition data with the features discovered by me
    """

    return data


def make_others_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return the competition data with the features discovered by others
    """

    return data


def join_all_features(xtr_my_features: pd.DataFrame,
                      xval_my_features: pd.DataFrame,
                      xtr_others_features: pd.DataFrame,
                      xval_others_features: pd.DataFrame,
                      ) -> pd.DataFrame:

    # return train, test
    pass
