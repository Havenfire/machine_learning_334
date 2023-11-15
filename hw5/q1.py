import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def normalize_feat(xTrain, xTest):
    scaler = StandardScaler()

    norm_xTrain = scaler.fit_transform(xTrain)
    norm_xTest = scaler.transform(xTest)

    return norm_xTrain, norm_xTest

def unreg_log(xTrain, yTrain, xTest, yTest):
    model = LogisticRegression(C=1e10)

    model.fit(xTrain, yTrain)

    # returns probability of yes
    y_probs = model.predict_proba(xTest)[:, 1]

    fpr, tpr, _ = roc_curve(yTest, y_probs)
    return fpr, tpr, auc(fpr, tpr)

def run_pca(xTrain, xTest):

    pca = PCA(n_components=None)
    pca.fit(xTrain)
    cumulative_variance_summation = np.cumsum(pca.explained_variance_ratio_)

    for index, val in enumerate(cumulative_variance_summation):
        if val > .95:
            num_components = index + 1
            break

    pca = PCA(n_components=num_components)

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
    pca_xTrain, pca_xTest, pca_components = run_pca(norm_xTrain, norm_xTest)

    fpr, tpr, auc_val = unreg_log(norm_xTrain, yTrain, norm_xTest, yTest)
    fpr_pca, tpr_pca, auc_val_pca = unreg_log(pca_xTrain, yTrain, pca_xTest, yTest)
    print(auc_val, auc_val_pca)

    plt.plot([0,1],[0,1], 'k--')
    plt.plot(fpr, tpr, label= "Normalized Logisitic Regression")
    plt.plot(fpr_pca, tpr_pca, label= "PCA logistic regression")
    plt.legend()
    plt.axis('square')

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title('ROC')
    plt.show()
if __name__ == "__main__":
    main()
