"""
This is a boilerplate pipeline 'FeatureEnginering'
generated using Kedro 0.18.1
"""
import pandas as pd
import numpy as np
from scipy import stats

def join_train_with_labels(train: pd.DataFrame, labels:pd.DataFrame) -> pd.DataFrame:
    # Do join based on id column
    train = train.merge(labels, on="customer_ID", how="left")
    
    # Return the data
    return train


def make_my_features(data: pd.DataFrame, target_col: str,  top_ratio: int) -> pd.DataFrame:
    """
    Returns the current dataframe with all personal feature enginering process
    """
    # Instantiate ratio cols making function
    def make_ratio_columns(data: pd.DataFrame, target_col: str, top_ratio: int) -> pd.DataFrame:
        """
        Returns the current dataframe with ratio (col_x/col_x) cols, this cols are made taking the "top_ratio" most correlated cols with the target
        and dividing them by the most negative correlated cols within the same target;
        """
        def bivariateTest(data, x, binarY):

            # Correlation betwen binary and continuous distributions functions
            def biserialCorr(data, x, binarY):
                # convert both variables to arrays
                xArray = np.array(data[x], dtype=int).ravel()
                binarYArray = np.array(data[binarY], dtype=int).ravel()
                
                # return biseralCorr
                return stats.pointbiserialr(binarYArray, xArray)

            def kruskalDiffTest(data, x, binarY):
                # take the two numpy arrays, one where the variable is one, and one where the variable is 0
                xClass_0 = np.array(data.query(f"{binarY} == 0")[x], dtype=int).ravel()
                xClass_1 = np.array(data.query(f"{binarY} == 1")[x], dtype=int).ravel()
                
                return stats.kruskal(xClass_0, xClass_1)

            # Make biserial correlation and kruskal hypthothesis test
            biserial = biserialCorr(data, x, binarY)
            kruskal = kruskalDiffTest(data, x, binarY)
            
            return biserial, kruskal

        correlations = {
            "Column": [],
            "BiCorr" : [],
            "Kruskal": []
        }

        # For every numeric column in dataframe columns
        numeric_cols  = [
            col for col in data.columns if str(data[col].dtype) != "object" and col != target_col
        ]
        for column in numeric_cols:

            # do the two statistics tests
            biseralCor, kruskal = bivariateTest(data=data, x=column, binarY=target_col)

            # append the results into relatory
            correlations["Column"].append(column)
            correlations["BiCorr"].append(biseralCor[0])
            correlations["Kruskal"].append(kruskal[0])

        # Make dataframe
        correlations = pd.DataFrame(correlations)

        # Order DataFrame
        correlations = correlations.sort_values(["BiCorr", "Kruskal"], ascending=[False, False])

        # Make Ratio columns
        ratioIteratiom = 0
        for rep in range(top_ratio):

            # Make current column name
            ratioName = "ratio_" + str(ratioIteratiom)

            # Take current positive col
            positiveCol = correlations["Column"].iloc[rep]

            # Take current negative col
            negativeCol = correlations["Column"].iloc[rep * -1]

            # New ratio col
            data[ratioName] = data[positiveCol] / data[negativeCol]

        return data

    # Use function
    data = make_ratio_columns(data=data, target_col=target_col, top_ratio=top_ratio)
    
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
