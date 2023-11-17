import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.stats import mode
from tqdm import tqdm

def generate_bootstrap(xTrain, yTrain):
    """
    Helper function to generate a bootstrap sample from the data. Each
    call should generate a different random bootstrap sample!

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of responses associated with training data.

    Returns
    -------
    xBoot : nd-array with shape n x d
        Bootstrap sample from xTrain
    yBoot : 1d array with shape n
        Array of responses associated with xBoot
    oobIdx : 1d array with shape k (which can be 0-(n-1))
        Array containing the out-of-bag sample indices from xTrain 
        such that using this array on xTrain will yield a matrix 
        with only the out-of-bag samples (i.e., xTrain[oobIdx, :]).
    """
    n = xTrain.shape[0]

    bootstrap_indices = np.random.choice(n, size=n, replace=True)
    
    xBoot = xTrain[bootstrap_indices]
    yBoot = yTrain[bootstrap_indices]

    oobIdx = [i for i in range(n) if i not in bootstrap_indices]

    return xBoot, yBoot, oobIdx


def generate_subfeat(xTrain, maxFeat):
    """
    Helper function to generate a subset of the features from the data. Each
    call is likely to yield different columns (assuming maxFeat is less than
    the original dimension)

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    maxFeat : int
        Maximum number of features to consider in each tree

    Returns
    -------
    xSubfeat : nd-array with shape n x maxFeat
        Subsampled features from xTrain
    featIdx: 1d array with shape maxFeat
        Array containing the subsample indices of features from xTrain
    """

    featIdx = np.random.choice(xTrain.shape[1], size=maxFeat, replace=False)
    xSubfeat = xTrain[:,featIdx]
    return xSubfeat, featIdx


class RandomForest(object):
    nest = 0           # number of trees
    maxFeat = 0        # maximum number of features
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    model = {}         # keeping track of all the models developed, where
                       # the key is the bootstrap sample. The value should be a dictionary
                       # and have 2 keys: "tree" to store the tree built
                       # "feat" to store the corresponding featIdx used in the tree


    def __init__(self, nest, maxFeat, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        nest: int
            Number of trees to have in the forest
        maxFeat: int
            Maximum number of features to consider in each tree
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.nest = nest
        self.maxFeat = maxFeat
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample

    def train(self, xFeat, y):
        stats = {}
        oobarr = np.full((self.nest, xFeat.shape[0]), False, dtype=bool)

        for i in range(self.nest):
            xBoot, yBoot, oobIdx = generate_bootstrap(xFeat, y)

            xSubfeat, featIdx = generate_subfeat(xBoot, self.maxFeat)
            dtc = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.maxDepth, min_samples_leaf=self.minLeafSample)
            dtc.fit(xSubfeat, yBoot)
            self.model[i] = {'tree': dtc, 'feat': featIdx}

            oob = xFeat[oobIdx, :]
            oob = oob[:, featIdx]

            oobarr[i, oobIdx] = True

            preds = np.zeros((xFeat.shape[0], i + 1))

            for j in range(i + 1):
                if np.any(oobarr[j]):
                    featIdx = self.model[j]['feat']
                    dt = self.model[j]['tree']
                    xSubfeat = xFeat[:, featIdx]
                    preds[:, j] = dt.predict(xSubfeat)


            non_zero_rows = np.any(preds != 0, axis=1)
            yhat = mode(preds[non_zero_rows], axis=1, keepdims=True).mode.flatten()
            yTrue = y[non_zero_rows]

            oob_error = 1.0 - accuracy_score(yTrue, yhat)
            stats[i] = oob_error

        return stats

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted response per sample
        """

        yHat = np.zeros(xFeat.shape[0])

        for i in range(self.nest):
            featIdx = self.model[i]["feat"]
            tree = self.model[i]["tree"]

            xSmall = xFeat[:, featIdx]
            yHat += tree.predict(xSmall)

        yHat = (np.round(yHat / self.nest)).astype(int)
        return yHat


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    """
    nest: Any,
    maxFeat: int,
    criterion: str,
    maxDepth: int,
    minLeafSample: int
    """

    # nest_values = range(1, 60)
    # maxFeatures = range(1, 12)
    # maxDepth = range(5, 100)
    # min_leaf_sample = range(2, 50)

    # accuracy_scores = []
    # for val in tqdm(maxDepth, desc="Testing nests"):
    #     model = RandomForest(50, 5, "gini", val, 2)
    #     model.train(xTrain, yTrain)
    #     yHat = model.predict(xTest)
    #     accuracy_scores.append(accuracy_score(yTest, yHat))

    # # Plot the accuracy scores
    # plt.plot(maxDepth, accuracy_scores, marker='o')
    # plt.xlabel('maxDepth')
    # plt.ylabel('Test Accuracy')
    # plt.title('Effect of maxDepth on Test Accuracy')
    # plt.show()

    model = RandomForest(35, 5, "gini", 25, 2)  
    trainStats = model.train(xTrain, yTrain)
    yHat = model.predict(xTest)
    print(1- accuracy_score(yTest, yHat))
    print(trainStats)

if __name__ == "__main__":
    main()