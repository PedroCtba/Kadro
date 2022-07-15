"""
This is a boilerplate pipeline 'FeatureEnginering'
generated using Kedro 0.18.1
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
import lightgbm as lgb


def train_cleaning_and_imputing(train: pd.DataFrame, train_labels: pd.DataFrame) -> pd.DataFrame:
    # Join labels with train
    train = train.merge(train_labels, on="customer_ID", how="left")
    del train_labels

    # Drop more than 70% null cols
    train = train.loc[:, train.isnull().mean() < .7]

    # Select cat cols
    numCols = train._get_numeric_data().columns
    catCols = list(set(train.columns) - set(numCols))
    filteredCatCols = list(set(train[catCols]) - {"S_2", "customer_ID"})

    # Use simple imputeron cat cols
    si = SimpleImputer(strategy="most_frequent")
    tr_train = pd.DataFrame(si.fit_transform(train[filteredCatCols]), columns = filteredCatCols)
    train[filteredCatCols] = tr_train[filteredCatCols]
    del tr_train
    
    # Take all nan count in numeric cols
    numericalCols = train.select_dtypes(np.number).columns
    nullSeries = train[numericalCols].isnull().sum()

    # Take columns to train and columns to fill null"s
    noneNullCols = nullSeries.loc[nullSeries == 0].index.tolist()
    noneNullCols.remove("target")

    # noneNullCols.remove("target")
    nullCols =  nullSeries.loc[nullSeries > 0].index.tolist()
    
    # Fillna NA lgb
    def fillna_with_lgb(data:pd.DataFrame, train_set_cols: list, col_to_fill: str) -> pd.DataFrame:
        # Instantiate lightgbm
        lg = lgb.LGBMRegressor(max_depth=-1, learning_rate=0.1, n_estimators=300)

        # Filter dataframe where variable to fill is not null
        dataNotNull = data.loc[ ~ data[col_to_fill].isnull()]

        # Make X and Y
        X = dataNotNull[train_set_cols]
        Y = dataNotNull[col_to_fill]
        del dataNotNull

        # Train the model
        lg.fit(X, Y)

        # Predict null values
        X_VAL = data[train_set_cols].loc[data[col_to_fill].isnull()]
        data.loc[data[col_to_fill].isnull(), col_to_fill] = lg.predict(X_VAL)

        return data
    
    # Do null filling for every column
    for nullC in nullCols:
        train = fillna_with_lgb(data=train, train_set_cols=noneNullCols, col_to_fill=nullC)

    # Groupy by customer
    train = train.groupby(['customer_ID']).nth(-1).reset_index(drop=True)

    # Transform ncat to numeric
    cols = ["D_63", "D_64", "D_68", "B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126"]
    train[cols] = train[cols].apply(pd.to_numeric, errors='coerce')

    # Drop unnecessary columns
    train.drop("S_2", axis=1, inplace=True)

    return train


def test_cleaning_and_imputing(test: pd.DataFrame) -> pd.DataFrame:
    # Drop more than 70% null cols
    test = test.loc[:, test.isnull().mean() < .7]

    # Select cat cols
    numCols = test._get_numeric_data().columns
    catCols = list(set(test.columns) - set(numCols))
    filteredCatCols = list(set(test[catCols]) - {"S_2", "customer_ID"})

    # Use simple imputeron cat cols
    si = SimpleImputer(strategy="most_frequent")
    tr_test = pd.DataFrame(si.fit_transform(test[filteredCatCols]), columns = filteredCatCols)
    test[filteredCatCols] = tr_test[filteredCatCols]
    del tr_test
    
    # Take all nan count in numeric cols
    numericalCols = test.select_dtypes(np.number).columns
    nullSeries = test[numericalCols].isnull().sum()

    # Take columns to test and columns to fill null"s
    noneNullCols = nullSeries.loc[nullSeries == 0].index.tolist()

    # noneNullCols.remove("target")
    nullCols =  nullSeries.loc[nullSeries > 0].index.tolist()
    
    # Fillna NA lgb
    def fillna_with_lgb(data:pd.DataFrame, test_set_cols: list, col_to_fill: str) -> pd.DataFrame:
        # Instantiate lightgbm
        lg = lgb.LGBMRegressor(max_depth=-1, learning_rate=0.1, n_estimators=300)

        # Filter dataframe where variable to fill is not null
        dataNotNull = data.loc[ ~ data[col_to_fill].isnull()]

        # Make X and Y
        X = dataNotNull[test_set_cols]
        Y = dataNotNull[col_to_fill]
        del dataNotNull
        
        # test the model
        lg.fit(X, Y)

        # Predict null values
        X_VAL = data[test_set_cols].loc[data[col_to_fill].isnull()]
        data.loc[data[col_to_fill].isnull(), col_to_fill] = lg.predict(X_VAL)

        return data
    
    # Do null filling for every column
    for nullC in nullCols:
        test = fillna_with_lgb(data=test, test_set_cols=noneNullCols, col_to_fill=nullC)

    # Groupy by customer
    test = test.groupby(['customer_ID']).nth(-1).reset_index(drop=False)

    # Transform ncat to numeric
    cols = ["D_63", "D_64", "D_68", "B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126"]
    test[cols] = test[cols].apply(pd.to_numeric, errors='coerce')

    # Drop unnecessary columns
    test.drop("S_2", axis=1, inplace=True)
    
    return test


def define_fe_process_with_train(cleaned_train: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Returns a correlation relatory that does biserial test correlation together with kruskal diff test
    """
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
            "Kruskal_st": [],
            "Kruskal_pvalue": []
        }

        # For every numeric column in dataframe columns
        numeric_cols  = list(data._get_numeric_data().columns)
        numeric_cols.remove(target_col)

        for column in numeric_cols:
            try:
                # do the two statistics tests
                biseralCor, kruskal = bivariateTest(data=data, x=column, binarY=target_col)

                # append the results into relatory
                correlations["Column"].append(column)
                correlations["BiCorr"].append(biseralCor[0])
                correlations["Kruskal_st"].append(kruskal[0])
                correlations["Kruskal_pvalue"].append(kruskal[1])

            except:
                pass

        # Make dataframe
        correlations = pd.DataFrame(correlations)

        # Order DataFrame
        correlations = correlations.sort_values(["BiCorr", "Kruskal_st"], ascending=[False, False])

        # Add treshold
        statistic = stats.chi2.ppf(1-0.05, df=1)
        correlations["kruskal_ppf"] = statistic

        # Evalute aception or not of H0
        correlations["kruskal_reject_h0"] = 0
        correlations.loc[correlations["Kruskal_st"] > correlations["kruskal_ppf"], "kruskal_reject_h0"] = 1

        return correlations

    # Make bivariate test
    bivariateRelatory = return_bivariate_test(data=cleaned_train, target_col=target_col)

    return bivariateRelatory


def make_my_features_at_train(cleaned_train: pd.DataFrame, correlationRelatory: pd.DataFrame, top_ratio: int) -> pd.DataFrame:
    """
    Returns the train dataframe with all personal feature engineering process
    """
    # Filter the dataset based on kruskal test aception of H0
    colsToDrop = correlationRelatory["Column"].loc[correlationRelatory["kruskal_reject_h0"] == 0].tolist()
    for col in colsToDrop: cleaned_train.drop(col, axis=1, inplace=True)
    
    # Make ratio columns
    for rep in range(top_ratio):
        if rep == 0:
            correlationPlus = correlationRelatory["Column"].iloc[rep]
            correlationMinus = correlationRelatory["Column"].iloc[-1]

        else:
            correlationPlus = correlationRelatory["Column"].iloc[rep]
            negRep = (rep + 1) * -1
            correlationMinus = correlationRelatory["Column"].iloc[negRep]
        
        # make ratio col
        ratioName = "ratio_" + correlationPlus + "_" + correlationMinus
        cleaned_train[ratioName] = cleaned_train[correlationPlus] / cleaned_train[correlationMinus]

    # Return cleaned train
    return cleaned_train


def make_my_features_at_test(cleaned_test: pd.DataFrame,  correlationRelatory: pd.DataFrame, top_ratio:int) -> pd.DataFrame:
    """
    Returns the test dataframe with all personal feature engineering process
    """
    # Filter the dataset based on kruskal test aception of H0
    colsToDrop = correlationRelatory["Column"].loc[correlationRelatory["kruskal_reject_h0"] == 0].tolist()
    for col in colsToDrop: cleaned_test.drop(col, axis=1, inplace=True)

    # Make ratio columns
    for rep in range(top_ratio):
        if rep == 0:
            correlationPlus = correlationRelatory["Column"].iloc[rep]
            correlationMinus = correlationRelatory["Column"].iloc[-1]

        else:
            correlationPlus = correlationRelatory["Column"].iloc[rep]
            negRep = (rep + 1) * -1
            correlationMinus = correlationRelatory["Column"].iloc[negRep]
        
        # make ratio col
        ratioName = "ratio_" + correlationPlus + "_" + correlationMinus
        cleaned_test[ratioName] = cleaned_test[correlationPlus] / cleaned_test[correlationMinus]

    # Return partitioned data
    return cleaned_test
