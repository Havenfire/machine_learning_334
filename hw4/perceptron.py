import argparse
import numpy as np
import pandas as pd
import time
import q1

class Perceptron(object):
    mEpoch = 1000  # maximum epoch size
    w = None       # weights of the perceptron

    def __init__(self, epoch):
        self.mEpoch = epoch

    def train(self, xFeat, y):
        """
        Train the perceptron using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the epochs and values the number of mistakes
        """
        stats = {}
        self.w = np.zeros(xFeat.shape[1])

        for epoch in range(self.mEpoch):
            mistakes = 0
            for xi, yi in zip(xFeat, y):
                self.w, mistake = self.sample_update(xi, yi)
                mistakes += mistake
            stats[epoch] = mistakes
        
        return stats

    def sample_update(self, xi, yi):
        """
        Given a single sample, give the resulting update to the weights

        Parameters
        ----------
        xi : numpy array of shape 1 x d
            Training sample 
        y : single value (-1, +1)
            Training label

        Returns
        -------
            wnew: numpy 1d array
                Updated weight value
            mistake: 0/1 
                Was there a mistake made 
        """

        prediction = np.dot(xi, self.w)
        if prediction * yi <= 0:
            self.w += yi * xi
            return self.w, 1
        return self.w, 0


    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted response per sample
        """
        yHat = []
        for row in xFeat:

            prediction = np.dot(row, self.w)
            if prediction >= 0:
                yHat.append(1)
            else:
                yHat.append(-1)

        return yHat



def transform_y(y):
    """
    Given a numpy 1D array with 0 and 1, transform the y 
    label to be -1 and 1

    Parameters
    ----------
    y : numpy 1-d array with labels of 0 and 1
        The true label.      

    Returns
    -------
    y : numpy 1-d array with labels of -1 and 1
        The true label but 0->-1
    """

    new_y = np.where(y == 0, -1, 1)

    return new_y

def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """
    err = 0
    for x1, x2 in zip(yHat, yTrue):
        if x1 != x2:
            err += 1

    
    

    return err


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()



def tune_perceptron(trainx, trainy, epochList):
    """
    Tune the preceptron to find the optimal number of epochs

    Parameters
    ----------
    trainx : a nxd numpy array
        The input from either binary / count matrix
    trainy : numpy 1d array of shape n
        The true label.    
    epochList: a list of positive integers
        The epoch list to search over  

    Returns
    -------
    epoch : int
        The optimal number of epochs
    """
    return 0


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

    # transform to -1 and 1
    yTrain = transform_y(yTrain)
    yTest = transform_y(yTest)

    np.random.seed(args.seed)   
    model = Perceptron(args.epoch)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    # print out the number of mistakes
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat, yTest))


if __name__ == "__main__":
    main()