import argparse
import numpy as np
import pandas as pd


class Knn(object):
    k = 0    # number of neighbors to use
    d = 0
    n = 0
    saved_xFeat = np.ndarray
    labels = np.ndarray


    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k

    def train(self, xFeat, y):
        """
        Train the k-nn model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        # TODO do whatever you need
        self.n = xFeat.size
        self.d = xFeat.ndim
        self.saved_xFeat = xFeat.copy()  # Make a copy as a numpy array
        self.labels = y.copy() 

        return self



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
            Predicted class label per sample
        """
        yHat = [] # variable to store the estimated class label

        if isinstance(xFeat, np.ndarray):
            xFeat = pd.DataFrame(data=xFeat)
        if isinstance(self.saved_xFeat, np.ndarray):
            self.saved_xFeat = pd.DataFrame(data=self.saved_xFeat)

        for val in xFeat.values:
            distances = []
            count = [0] * self.d
            for val2 in self.saved_xFeat.values:
                distances.append(np.linalg.norm(val - val2))
            
            sorted_distances_index = np.argsort(distances) #this should be indicies
            for i in range(0, self.k):
                count[int(self.labels[sorted_distances_index[i]])] += 1
            m = max(count)
            yHat.append(count.index(m))

        return yHat


def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    # TODO calculate the accuracy
    acc = 0
    correct = 0

    for val, val2 in zip(yHat, yTrue):
        if val == val2:
            correct += 1

    acc = correct / len(yHat)

    return acc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
