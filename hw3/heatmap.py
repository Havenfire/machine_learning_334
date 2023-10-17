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
    corrMat_list = []

    # Iterate through the columns of the dataframe
    for col1 in df.columns:
        row = []
        for col2 in df.columns:
            # Calculate the Pearson correlation coefficient
            corr = stats.pearsonr(df[col1], df[col2]).correlation
            if corr >=.5:
                row.append(1)
            elif corr <= -.5:
                row.append(-1)
            else:        
                row.append(0)


        corrMat_list.append(row)

    # Create a new dataframe using the correlation values and column names
    corrMat = pd.DataFrame(corrMat_list, columns=df.columns)

    # Plot the correlation matrix as a heatmap
    plt.subplots(figsize=(20,20))

    hm = sn.heatmap(data=corrMat,  cmap=sn.diverging_palette(20, 220, n=200),)
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
    print(vertical_concat.head())
    cal_corr(vertical_concat)

if __name__ == "__main__":
    main()
