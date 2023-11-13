import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

def normalize_feat(xTrain, xTest):
    scaler = StandardScaler()

    norm_xTrain = scaler.fit_transform(xTrain)
    norm_xTest = scaler.transform(xTest)

    return norm_xTrain, norm_xTest

def unreg_log(xTrain, yTrain, xTest, yTest):
    model = LogisticRegression(C= 1e10)

    model.fit(xTrain, yTrain)

    y_probs = model.predict(xTest)

    fpr, tpr, _ = roc_curve(yTest, y_probs)
    return fpr, tpr, auc(fpr, tpr)

def run_pca(xTrain, xTest):

    pca = PCA(n_components = 9)

    t_xTrain = pca.fit_transform(xTrain)
    t_xTest = pca.transform(xTest)

    return t_xTrain, t_xTest, pca.components_

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain", default="xTrain.csv", help="filename for features of the training data")
    parser.add_argument("--yTrain", default="yTrain.csv", help="filename for labels associated with training data")
    parser.add_argument("--xTest", default="xTest.csv", help="filename for features of the test data")
    parser.add_argument("--yTest", default="yTest.csv", help="filename for labels associated with the test data")

    args = parser.parse_args()

    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)

    xTrain = xTrain.to_numpy()
    yTrain = yTrain.to_numpy().ravel()

    xTest = xTest.to_numpy()
    yTest = yTest.to_numpy().ravel()

    norm_xTrain, norm_xTest = normalize_feat(xTrain, xTest)

    unreg_log(xTrain, yTrain, xTest, yTest)
    TxTrain, TxTest, PCA_C = run_pca(norm_xTrain, norm_xTest)

if __name__ == "__main__":
    main()
