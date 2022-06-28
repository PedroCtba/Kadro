"""
This is a boilerplate pipeline 'FeatureEnginering'
generated using Kedro 0.18.1
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from scipy import stats

def train_cleaning_and_imputing(train: pd.DataFrame) -> pd.DataFrame:
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

    # Take all nan count in numeric cols
    numericalCols = train.select_dtypes(np.number).columns
    nullSeries = train[numericalCols].isnull().sum()

    # Take columns to train and columns to fill null"s
    noneNullCols = nullSeries.loc[nullSeries == 0].index.tolist()
    noneNullCols.remove("target")
    nullCols =  nullSeries.loc[nullSeries > 0].index.tolist()

    # Fillna NA lgbm
    def fillna_with_lgb(data:pd.DataFrame, train_set_cols: list, col_to_fill: str) -> pd.DataFrame:
        # Instantiate lightgbm
        lg = lgb.LGBMRegressor(max_depth=-1, learning_rate=0.1, n_estimators=300)

        # Filter dataframe where variable to fill is not null
        dataNotNull = data.loc[~data[col_to_fill].isnull()]

        # Make X and Y
        X = dataNotNull[train_set_cols]
        Y = dataNotNull[col_to_fill]

        # Train the model
        lg.fit(X, Y)

        # Predict null values
        X_VAL = data[train_set_cols].loc[data[col_to_fill].isnull()]
        data.loc[data[col_to_fill].isnull(), col_to_fill] = lg.predict(X_VAL)

        return data
    
    # Do null filling for every column
    for nullC in nullCols:
        train = fillna_with_lgb(data=train, train_set_cols=noneNullCols, col_to_fill=nullC)

    #Fixing date columns
    train["S_2_day"] = train["S_2"].dt.day
    train["S_2_month"] = train["S_2"].dt.month
    train["S_2_year"] = train["S_2"].dt.year

    # Groupy by customer
    train = train.groupby(['customer_ID']).nth(-1).reset_index(drop=True)

    # Transform ncat to numeric
    cols = ["D_63", "D_64", "D_68", "B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126"]
    train[cols] = train[cols].apply(pd.to_numeric, errors='coerce')

    # Drop unnecessary columns
    train.drop("S_2", axis=1, inplace=True)

    return train


def make_my_features(cleaned_train: pd.DataFrame, target_col: str,  top_ratio: int) -> pd.DataFrame:
    """
    Returns the current dataframe with all feature enginerring process
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
            "Kruskal": []
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
                correlations["Kruskal"].append(kruskal[0])

            except:
                pass

        # Make dataframe
        correlations = pd.DataFrame(correlations)

        # Order DataFrame
        correlations = correlations.sort_values(["BiCorr", "Kruskal"], ascending=[False, False])

        return correlations

    # Make bivariate test
    bivariateRelatory = return_bivariate_test(data=cleaned_train, target_col=target_col)
    
    # Make ratio columns
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
        cleaned_train[ratioName] = cleaned_train[correlationPlus] / cleaned_train[correlationMinus]

    # Return partitioned data
    return cleaned_train


def use_scalers_based_on_outliers(fe_train: pd.DataFrame) -> pd.DataFrame:
    # Take all the numerical columns
    numericalColumns = [col for col in fe_train.columns if fe_train[col].dtype == np.number]

    # Find wich cols have a good number of outliers
    robustScaler = []
    minMaxScaler = []

    for ncol in numericalColumns:
        # Calculate Q1 and Q3
        Q1 = fe_train[ncol].quantile(0.25)
        Q3 = fe_train[ncol].quantile(0.75)

        # Count the number of outliers
        if len(fe_train[ncol].loc[(fe_train[ncol] < Q1) | (fe_train[ncol] > Q3)]) > 2:
            robustScaler.append(ncol)
        else:
            minMaxScaler.append(ncol)
    
    # Substitue infinities with specific value
    fe_train.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Split train and test
    X = fe_train[[col for col in fe_train.columns if col != "target"]]
    Y = pd.DataFrame(fe_train["target"])
    del fe_train

    # Use robust scaler
    r = RobustScaler()
    X[robustScaler] = r.fit_transform(X[robustScaler])

    # Use min max scaler
    m = MinMaxScaler()
    X[minMaxScaler] = m.fit_transform(X[minMaxScaler])

    return X, Y

        
def join_all_features(xtr_my_features: pd.DataFrame,
                      xval_my_features: pd.DataFrame,
                      xtr_others_features: pd.DataFrame,
                      xval_others_features: pd.DataFrame,
                      ) -> pd.DataFrame:

    # return train, test
    pass

