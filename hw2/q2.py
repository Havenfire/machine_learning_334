import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import time
import sklearn.model_selection as ms
from sklearn.metrics import roc_auc_score


def holdout(model, xFeat, y, testSize):
    """
    Split xFeat into random train and test based on the testSize and
    return the model performance on the training and test set. 

    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : numpy.nd-array with shape n x d
        Features of the dataset 
    y : numpy.1d-array with shape n 
        Labels of the dataset
    testSize : float
        Portion of the dataset to serve as a holdout. 

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

    # remember to change to random_state = NONE
    xTrain, xTest, yTrain, yTest = ms.train_test_split(xFeat, y, test_size=testSize, random_state=None)

    model.fit(xTrain, yTrain)

    trainPredictions = model.predict_proba(xTrain)[:, 1]
    testPredictions = model.predict_proba(xTest)[:, 1]

    trainAuc = roc_auc_score(yTrain, trainPredictions)
    testAuc = roc_auc_score(yTest, testPredictions)
    endTime = time.time()

    timeElapsed = endTime - startTime

    return trainAuc, testAuc, timeElapsed


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


def mc_cv(model, xFeat, y, testSize, s):
    """
    Evaluate the model using s samples from the
    Monte Carlo cross-validation approach where
    for each sample you split xFeat into
    random train and test based on the testSize.
    Returns the model performance on the training and
    test datasets.

    Parameters
    ----------
    model : sklearn.tree.DecisionTreeClassifier
        Decision tree model
    xFeat : numpy.nd-array with shape n x d
        Features of the dataset 
    y : numpy.1d-array with shape n
        Labels of the dataset
    testSize : float
        Portion of the dataset to serve as a holdout.
    s : int
        Number of Monte Carlo iterations

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    avg_train_auc = 0
    avg_test_auc = 0
    timeElapsed = 0

    cumulated_time = 0
    cumulated_train_auc = 0
    cumulated_test_auc = 0
    
    
    start_time = time.time()

    for i in range(0, s):
        # Split the data into train and test sets
        xTrain, xTest, yTrain, yTest = ms.train_test_split(xFeat, y, test_size=testSize)

        # Train the model on the training data
        model.fit(xTrain, yTrain)
        
        # Make predictions on training and test data
        yTrainPred = model.predict_proba(xTrain)[:, 1]
        yTestPred = model.predict_proba(xTest)[:, 1]

        # Calculate AUC for training and test data
        cumulated_train_auc += roc_auc_score(yTrain, yTrainPred)
        cumulated_test_auc += roc_auc_score(yTest, yTestPred)

    end_time = time.time()


    # Calculate average AUC and time elapsed
    avg_train_auc = cumulated_train_auc / s
    avg_test_auc = cumulated_test_auc / s
    timeElapsed = end_time - start_time

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
    # create the decision tree classifier
    dtClass = DecisionTreeClassifier(max_depth=15,
                                     min_samples_leaf=10)
    # use the holdout set with a validation size of 30 of training
    aucTrain1, aucVal1, time1 = holdout(dtClass, xTrain, yTrain, 0.30)
    # use 2-fold validation
    aucTrain2, aucVal2, time2 = kfold_cv(dtClass, xTrain, yTrain, 2)
    # use 5-fold validation
    aucTrain3, aucVal3, time3 = kfold_cv(dtClass, xTrain, yTrain, 5)
    # use 10-fold validation
    aucTrain4, aucVal4, time4 = kfold_cv(dtClass, xTrain, yTrain, 10)
    # use MCCV with 5 samples
    aucTrain5, aucVal5, time5 = mc_cv(dtClass, xTrain, yTrain, 0.30, 5)
    # use MCCV with 10 samples
    aucTrain6, aucVal6, time6 = mc_cv(dtClass, xTrain, yTrain, 0.30, 10)
    # train it using all the data and assess the true value
    trainAuc, testAuc = sktree_train_test(
        dtClass, xTrain, yTrain, xTest, yTest)
    perfDF = pd.DataFrame([['Holdout', aucTrain1, aucVal1, time1],
                           ['2-fold', aucTrain2, aucVal2, time2],
                           ['5-fold', aucTrain3, aucVal3, time3],
                           ['10-fold', aucTrain4, aucVal4, time4],
                           ['MCCV w/ 5', aucTrain5, aucVal5, time5],
                           ['MCCV w/ 10', aucTrain6, aucVal6, time6],
                           ['True Test', trainAuc, testAuc, 0]],
                          columns=['Strategy', 'TrainAUC', 'ValAUC', 'Time'])
    print(perfDF)


if __name__ == "__main__":
    main()
