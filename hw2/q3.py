import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import time
import sklearn.model_selection as ms
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

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

def kfold_cv(model, xFeat, y, k):
    """
    Split xFeat into k different groups, and then use each of the
    k-folds as a validation set, with the model fitting on the remaining
    k-1 folds. Return the model performance on the training and
    validation (test) set. 


    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : numpy.nd-array with shape n x d
        Features of the dataset 
    y : numpy.1d-array with shape n
        Labels of the dataset
    k : int
        Number of folds or groups (approximately equal size)

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    
    startTime = time.time()

    total_train_auc = 0
    total_test_auc = 0
    
    kf = ms.KFold(n_splits=k)

    for train_index, test_index in kf.split(xFeat):
        xTrain, xTest = xFeat[train_index], xFeat[test_index]
        yTrain, yTest = y[train_index], y[test_index]

        
        model.fit(xTrain, yTrain)
  
        trainPredictions = model.predict_proba(xTrain)[:, 1]
        testPredictions = model.predict_proba(xTest)[:, 1]

        trainAuc = roc_auc_score(yTrain, trainPredictions)
        testAuc = roc_auc_score(yTest, testPredictions)


        
        total_train_auc += trainAuc
        total_test_auc += testAuc

    avg_train_auc = total_train_auc / k
    avg_test_auc = total_test_auc / k
    
    endTime = time.time()
    timeElapsed = endTime - startTime

    return avg_train_auc, avg_test_auc, timeElapsed



def sktree_train_test(model, xTrain, yTrain, xTest, yTest):
    """
    Given a sklearn tree model, train the model using
    the training dataset, and evaluate the model on the
    test dataset.

    Parameters
    ----------
    model : DecisionTreeClassifier object
        An instance of the decision tree classifier 
    xTrain : numpy.nd-array with shape nxd
        Training data
    yTrain : numpy.1d array with shape n
        Array of labels associated with training data
    xTest : numpy.nd-array with shape mxd
        Test data
    yTest : numpy.1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    trainAUC : float
        The AUC of the model evaluated on the training data.
    testAuc : float
        The AUC of the model evaluated on the test data.
    """
    # fit the data to the training dataset
    model.fit(xTrain, yTrain)
    # predict training and testing probabilties
    yHatTrain = model.predict_proba(xTrain)
    yHatTest = model.predict_proba(xTest)
    # calculate auc for training
    fpr, tpr, thresholds = metrics.roc_curve(yTrain,
                                             yHatTrain[:, 1])
    trainAuc = metrics.auc(fpr, tpr)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(yTest,
                                             yHatTest[:, 1])
    testAuc = metrics.auc(fpr, tpr)
    return trainAuc, testAuc


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
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

    #hyper-paramters
    dtClass_original = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)
    dtClass_5_percent_removed = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)
    dtClass_10_percent_removed = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)
    dtClass_20_percent_removed = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)

    # use 5-fold validation

    KnnClass = KNeighborsClassifier(n_neighbors = 6)

    aucTrain_original, aucVal_original, time_original = kfold_cv(KnnClass, xTrain, yTrain, 5)
    xTrain_100, xTest_100, yTrain_100, yTest_100 = train_test_split(xTrain, yTrain, random_state=None)

    xTrain_95, _, yTrain_95, yTest_95 = train_test_split(xTrain, yTrain, test_size=0.05, random_state=None)


    aucTrain_95, aucVal_95, time_95 = kfold_cv(KnnClass, xTrain_95, yTrain_95, 5)

    xTrain_90, _, yTrain_90, yTest_90 = train_test_split(xTrain, yTrain, test_size=0.10, random_state=None)

    aucTrain_90, aucVal_90, time_90 = kfold_cv(KnnClass, xTrain_90, yTrain_90, 5)

    xTrain_80, X, yTrain_80, yTest_80 = train_test_split(xTrain, yTrain, test_size=0.20, random_state=None)



    aucTrain_80, aucVal_80, time_80 = kfold_cv(KnnClass, xTrain_80, yTrain_80, 5)


    trainAuc_original, testAuc_original = sktree_train_test(dtClass_original, xTrain, yTrain, xTest, yTest)

    score = dtClass_original.score(X, yTest_80)
    print(score)

    trainAuc_5_percent_removed, testAuc_5_percent_removed = sktree_train_test(dtClass_5_percent_removed, xTrain_95, yTrain_95, xTest, yTest)
    trainAuc_10_percent_removed, testAuc_10_percent_removed = sktree_train_test(dtClass_10_percent_removed, xTrain_90, yTrain_90, xTest, yTest)
    trainAuc_20_percent_removed, testAuc_20_percent_removed = sktree_train_test(dtClass_20_percent_removed, xTrain_80, yTrain_80, xTest, yTest)


    perfDF = pd.DataFrame([ ['KNN', aucTrain_original, aucVal_original, time_original],
                            ['KNN-95', aucTrain_95, aucVal_95, time_95],
                            ['KNN-90', aucTrain_90, aucVal_90, time_90],
                            ['KNN-80', aucTrain_80, aucVal_80, time_80],

                            ['DT Original', trainAuc_original, testAuc_original, 0],
                            ['DT 5 removed', trainAuc_5_percent_removed, testAuc_5_percent_removed, 0],
                            ['DT 10 removed', trainAuc_10_percent_removed, testAuc_10_percent_removed, 0],
                            ['DT 20 removed', trainAuc_20_percent_removed, testAuc_20_percent_removed, 0]],
                          columns=['Strategy', 'TrainAUC', 'ValAUC', 'Time'])
    print(perfDF)


if __name__ == "__main__":
    main()
