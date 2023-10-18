import argparse
import numpy as np
import pandas as pd
import time
import pdb
import matplotlib.pyplot as plt


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

    b = xi.shape[0]

    #what the model thinks the hypothesis should be


    #calculates which direction to shift the model to the target
    hypothesis = np.matmul(xi, beta)
    grad = (-2 / b) * (xi.T @ (yi - hypothesis))
    grad = grad.reshape(len(grad),-1)


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

        xTrain = np.concatenate([xTrain, np.ones((len(xTrain), 1))], axis=-1)
        xTest = np.concatenate([xTest, np.ones((len(xTest), 1))], axis=-1)

        # sample_size = int(0.4 * len(xTrain))
        # selected_indices = np.random.choice(len(xTrain), sample_size, replace=False)
        # xTrain = xTrain[selected_indices]
        # yTrain = yTrain[selected_indices]


        n_samples, n_features = xTrain.shape
        self.beta = np.zeros(n_features)
        total_iterations = 0

        trainStats = {}
        batch_num = n_samples // self.bs
        start = time.time()
        

        for epoch in range(0, self.mEpoch):

            if epoch == 0:
                self.beta = np.random.rand(xTrain.shape[1],1)
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            xTrain_shuffled = xTrain[indices]
            yTrain_shuffled = yTrain[indices]
           
            cumulative_derivative = np.zeros((xTrain.shape[1],1))

            for batch in range(batch_num): 
                start_i = batch * self.bs
                end_i = (batch + 1) * self.bs
                xi = xTrain_shuffled[start_i:end_i]
                yi = yTrain_shuffled[start_i:end_i]


                cumulative_derivative = cumulative_derivative + grad_pt(self.beta, xi, yi)
                self.beta = self.beta - self.lr * (cumulative_derivative)
                

            end = time.time()


            train_mse = self.mse(xTrain, yTrain)
            test_mse = self.mse(xTest, yTest)

           
            trainStats[total_iterations] = {
                    'time': end - start, 
                    'train-mse': train_mse,
                    'test-mse': test_mse,
                }
            print(total_iterations)
            total_iterations += 1

            
        df = pd.DataFrame.from_dict(trainStats, orient="index")
        plt.plot(df)
        plt.title("Learning rate of : 0.00000001", )
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.show()
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

