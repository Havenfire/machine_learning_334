import argparse
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sn
import matplotlib.pyplot as plt

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



    something = ['T3', 'T4', 'T5', 'T7', 'T8', 'T9', 'T_out', 'Tdewpoint', 'RH_6', 'RH_2', 'RH_3', 'RH_4', 'RH_7', 'RH_8', 'RH_9', 'year', 'Visibility']
    df.drop(something, inplace = True, axis = 0)
    df.drop(something, inplace = True, axis = 1)
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
    corrMat = corrMat.set_index(corrMat.columns)
    corrMat = corrMat.reindex(sorted(corrMat.columns), axis=1)
    corrMat = corrMat.sort_index()
    
    
    # Plot the correlation matrix as a heatmap
    plt.subplots(figsize=(20,20))

    hm = sn.heatmap(data=corrMat,  cmap=sn.diverging_palette(20, 220, n=200), vmin=-1, vmax=1)
    plt.show()

    return corrMat

def main():
    """
    Main file to run from the command line.
    """
    # Set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain", help="filename of the updated training data")
    parser.add_argument("outTest", help="filename of the updated test data")
    parser.add_argument("--trainFile", default="new_xTrain.csv", help="filename of the training data")
    parser.add_argument("--yFile", default="eng_yTrain.csv", help="filename of the test data")
    args = parser.parse_args()

    # Load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    yTrain = pd.read_csv(args.yFile)
    vertical_concat = pd.concat([xTrain, yTrain], axis=1)
    cal_corr(vertical_concat)

if __name__ == "__main__":
    main()
