import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def calculate_split_score(y, criterion):
    """
    Given a numpy array of labels associated with a node, y, 
    calculate the score based on the crieterion specified.

    Parameters
    ----------
    y : numpy.1d array with shape n
        Array of labels associated with a node
    criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
    Returns
    -------
    score : float
        The gini or entropy associated with a node
    """



    values, counts = np.unique(y, return_counts=True)
    total_counts = len(y)
    if criterion == "gini":
        gini_score = 0.0
        for num in counts:
            probability = num / total_counts
            gini_score += probability * (1 - probability)
        return gini_score

    if criterion == "entropy":
        entropy_score = 0.0
        for num in counts:
            probability = num / total_counts
            entropy_score -= probability * np.log2(probability)
        return entropy_score
    return -1


class Node(object):
    data = None
    depth = 0
    left = None
    right = None
    split_point = ()

    def __init__(self, depth, data: np.ndarray):
        self.depth = depth
        self.data = data

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right
    
    def get_split_point(self):
        return self.split_point
    
    def set_split_point(self, best_point):
        self.split_point = best_point


class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    root_node = None

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample

    def train(self, xFeat, y):
        """
        Train the decision tree model.

        Parameters
        ----------
        xFeat : numpy.nd-array with shape n x d
            Training data 
        y : numpy.1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """

        data = np.column_stack([xFeat, y])
        self.root_node = Node(depth=0, data=data)
        self.recursive_train(self.root_node)
        return self

    def recursive_train(self, current_node: Node):

        # stopping conditions
        if current_node == None:
            return
        if current_node.data.shape[0] < self.minLeafSample:
            return
        if current_node.depth > self.maxDepth:
            return
        if current_node.data.size == 0:
            return
        if calculate_split_score(current_node.data[:, -1], self.criterion) == 0:
            return

        y_values = current_node.data[:, -1]
        if len(np.unique(y_values)) == 1:  # All labels are the same
            return
        
        best_split_location = None
        best_split_value = float('inf')
        left_data = None
        right_data = None


        #finding best split condition
        for column in range(0, current_node.data.shape[1] - 1):
            for row in range(0, current_node.data.shape[0]):
                left_data = current_node.data[current_node.data[:, column] < current_node.data[row, column]]
                right_data = current_node.data[current_node.data[:, column] >= current_node.data[row, column]]

                split_value_left = calculate_split_score(left_data[:, -1], self.criterion)
                split_value_right = calculate_split_score(right_data[:, -1], self.criterion)

                current_split_value = split_value_left * len(left_data) / len(current_node.data) + split_value_right * len(right_data) / len(current_node.data)
                
                if (current_split_value < best_split_value):
                    best_split_location = (column, current_node.data[row, column])
                    best_split_value = current_split_value
        # print(best_split_location)
        
        left_data = current_node.data[current_node.data[:,best_split_location[0]] < best_split_location[1]]
        right_data = current_node.data[current_node.data[:,best_split_location[0]] >= best_split_location[1]]

        current_node.set_split_point(best_split_location)

        current_node.left = Node(depth=current_node.depth + 1, data=left_data)
        current_node.right = Node(depth=current_node.depth + 1, data=right_data)

        self.recursive_train(current_node.left)
        self.recursive_train(current_node.right)

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : numpy.nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : numpy.1d array with shape m
            Predicted class label per sample
        """
        yHat = np.array([])  # variable to store the estimated class label

        for row in range(0, xFeat.shape[0]):
            yHat = np.append(yHat, self.predict_recursion(xFeat[row], self.root_node))

        return yHat
    
    def predict_recursion(self, row, current_node: Node):

        if current_node.left == None and current_node.right == None:
            last_column = current_node.data[:, -1].astype(int)

            return np.bincount(last_column).argmax()
        else:
            split_attribute, split_value = current_node.split_point

            if row[split_attribute] < split_value:
                return self.predict_recursion(row, current_node.left)
            else:
                return self.predict_recursion(row, current_node.right)

        


def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : numpy.nd-array with shape n x d
        Training data 
    yTrain : numpy.1d array with shape n
        Array of labels associated with training data.
    xTest : numpy.nd-array with shape m x d
        Test data 
    yTest : numpy.1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain)
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain, yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest, yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
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
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()
    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
