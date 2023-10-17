import argparse
import numpy as np
import pandas as pd
from scipy import stats 
import seaborn as sn 
import matplotlib.pyplot as plt 
from sklearn import preprocessing

def extract_features(df):
    """
    Given a pandas dataframe, extract the relevant features
    from the date column

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with the new features
    """

    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y %H:%M')

    # Extract the desired features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek    
    df['hour'] = df['date'].dt.hour
    
    df = df.drop(columns=['date'])

    return df


def cal_corr(df):
    """
    Given a pandas dataframe (include the target variable at the last column),
    calculate the correlation matrix (compute pairwise correlation of columns)

    Parameters
    ----------
    df : pandas dataframe
        Training or test data (with the target variable)
    Returns
    -------
    corrMat : pandas dataframe
        Correlation matrix
    """
    # Create an empty list to store correlation values
    corrMat_list = []

    # Iterate through the columns of the dataframe
    for col1 in df.columns:
        row = []
        for col2 in df.columns:
            # Calculate the Pearson correlation coefficient
            corr = stats.pearsonr(df[col1], df[col2]).correlation
            row.append(corr)

        corrMat_list.append(row)

    # Create a new dataframe using the correlation values and column names
    corrMat = pd.DataFrame(corrMat_list, columns=df.columns)

    return corrMat


#remove all values that are highly correlated
def select_features(df):
    """
    Select the features to keep based on correlation with a threshold

    Parameters:
    df : pandas dataframe
        Training or test data

    Returns:
    df : pandas dataframe
        The updated dataframe with a subset of the columns
    """

    # Calculate the correlation matrix
    features_to_remove = ['T3', 'T4', 'T5', 'T7', 'T8', 'T9', 'T_out', 'Tdewpoint', 'RH_6', 'RH_2', 'RH_3', 'RH_4', 'RH_7', 'RH_8', 'RH_9', 'year', 'Visibility']
    df.drop(features_to_remove, inplace = True, axis = 1)

    return df


def preprocess_data(trainDF, testDF):
    """
    Preprocess the training data and testing data

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data 
    testDF : pandas dataframe
        Test data 
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """

    mean = trainDF.mean()
    std = trainDF.std()

    normalized_trainDF = (trainDF - mean) / std
    normalized_testDF = (testDF - mean) / std


    return normalized_trainDF, normalized_testDF


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)
    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest = extract_features(xTest)
    
    # select the features
    xNewTrain = select_features(xNewTrain)
    xNewTest = select_features(xNewTest)

    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)

    # save it to csv
    xTrainTr.to_csv(args.outTrain, index=False)
    xTestTr.to_csv(args.outTest, index=False)

if __name__ == "__main__":
    main()
