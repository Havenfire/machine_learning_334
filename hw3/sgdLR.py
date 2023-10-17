import argparse
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy


def grad_pt(beta, xi, yi):
    """
    Calculate the gradient for a mini-batch sample.

    Parameters
    ----------
    beta : 1d array with shape d
    xi : 2d numpy array with shape b x d
        Batch training data
    yi: 2d array with shape bx1
        Array of responses associated with training data.

    Returns
    -------
        grad : 1d array with shape d
    """

    #what the model thinks the hypothesis should be
    h = np.dot(xi, beta)
    #calculates which direction to shift the model to the target
    grad = np.dot(xi.transpose(), (h - yi))
    return grad


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000 # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        # TODO: DO SGD
        
        n_samples, n_features = xTrain.shape
        self.beta = np.zeros(n_features)  # Initialize beta with zeros
        total_iterations = 0

        trainStats = {}
        batch_num = n_samples // self.bs
        start = time.time()

        for epoch in range(1, self.mEpoch + 1):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            xTrain_shuffled = xTrain[indices]
            yTrain_shuffled = yTrain[indices]

            for batch in range(batch_num): 
                start = batch * self.bs
                end = (batch + 1) * self.bs
                xi = xTrain_shuffled[start:end]
                yi = yTrain_shuffled[start:end]

                self.beta = grad_pt(self.beta, xi, yi)
                
                total_iterations += 1
            end = time.time()

            # print(xTrain, "\n\n", yTrain)
            # print(xTrain.shape, "\n\n", yTrain.shape)
            trainStats[total_iterations] = {
                'time': end - start, 
                'train-mse': self.mse(xTrain, yTrain),
                'test-mse': self.mse(xTest, yTest)
            }

        return trainStats


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
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # setting the seed for deterministic behavior
    np.random.seed(args.seed)   
    model = SgdLR(args.lr, args.bs, args.epoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()

