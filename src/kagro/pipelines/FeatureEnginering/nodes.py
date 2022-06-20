"""
This is a boilerplate pipeline 'FeatureEnginering'
generated using Kedro 0.18.1
"""
import pandas as pd
from typing import Any, Callable, Dict
import numpy as np
from scipy import stats

def join_train_with_labels(partitioned_data: Dict[str, Callable[[], Any]], labels:pd.DataFrame) -> Dict[str, Any]:
    # Saving dictionary
    SaveDictionary = {}

    # Do join based on ID column
    for data_path, data_load_func in sorted(partitioned_data.items()):
        # Load using load function
        data = data_load_func()

        # Do merge
        data = data.merge(labels, on="customer_ID", how="left")

        # Append to dictionary using data path no make name
        keyName = "joined_" + data_path.split("/")[-1].split(".")[0]
        SaveDictionary[keyName] = data

        # Clean memory
        del data
    
    # Return partitioned data
    return SaveDictionary


def make_my_features(partitioned_data: Dict[str, Callable[[], Any]], target_col: str,  top_ratio: int) -> Dict[str, Any]:
    """
    Returns the current dataframe with all personal feature enginering process
    """
    # Saving dictionary
    SaveDictionary = {}

    # Instantiate ratio cols making function
    def return_bivariate_test(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
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
            try:
                # do the two statistics tests
                biseralCor, kruskal = bivariateTest(data=data, x=column, binarY=target_col)

                # append the results into relatory
                correlations["Column"].append(column)
                correlations["BiCorr"].append(biseralCor[0])
                correlations["Kruskal"].append(kruskal[0])

            except:
                pass

        # Make dataframe
        correlations = pd.DataFrame(correlations)

        # Order DataFrame
        correlations = correlations.sort_values(["BiCorr", "Kruskal"], ascending=[False, False])

        return correlations

    # Concat the data to make bivariate test
    concatenedData = pd.DataFrame()

    for partition_key, partition_load_func in sorted(partitioned_data.items()):
        # Load with loading function
        partitionedData = partition_load_func()

        # Concat the data
        concatenedData = pd.concat([concatenedData, partitionedData], ignore_index=True)

        #  Clean memory
        del partitionedData

    # Make bivariate test
    bivariateRelatory = return_bivariate_test(data=concatenedData, target_col=target_col)

    # Clean memory
    del concatenedData

    # Use top "x" features to make ratio on all training data
    for data_path, data_load_func in sorted(partitioned_data.items()):
        # Load using load function
        data = data_load_func()

        # Make ratio
        for rep in range(top_ratio):
            if rep == 0:
                correlationPlus = bivariateRelatory["Column"].iloc[rep]
                correlationMinus = bivariateRelatory["Column"].iloc[-1]
            else:
                correlationPlus = bivariateRelatory["Column"].iloc[rep]
                negRep = (rep + 1) * -1
                correlationMinus = bivariateRelatory["Column"].iloc[negRep]
            
            # make ratio col
            ratioName = "ratio_" + correlationPlus + "_" + correlationMinus
            data[ratioName] = data[correlationPlus] / data[correlationMinus]

        # Append to dictionary using data path no make name
        keyName = "feature_enginered_" + data_path.split("/")[-1].split(".")[0]
        SaveDictionary[keyName] = data

        # Clean memory
        del data
   
    # Return partitioned data
    return SaveDictionary


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

