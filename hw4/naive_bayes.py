import numpy as np
import argparse
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Assuming you have two datasets: X_train, y_train, X_test, and y_test.
# X_train contains the text messages, and y_train contains their labels (0 for non-spam, 1 for spam).
# X_test contains the test text messages, and y_test contains their true labels.

def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)


    # mnb = MultinomialNB()
    # mnb.fit(xTrain, yTrain)
    # mnb_predictions = mnb.predict(xTest)
    # mnb_accuracy = accuracy_score(yTest, mnb_predictions)
    # mnb_mistakes = calc_mistakes(yTest, mnb_predictions)
    # print(f"Accuracy: {mnb_accuracy:.2f}")
    # print(f"Mistakes: {mnb_mistakes}")

    # bnb = BernoulliNB()
    # bnb.fit(xTrain, yTrain)
    # bnb_predictions = bnb.predict(xTest)
    # bnb_accuracy = accuracy_score(yTest, bnb_predictions)
    # bnb_mistakes = calc_mistakes(yTest, bnb_predictions)
    # print(f"Accuracy: {bnb_accuracy:.2f}")
    # print(f"Mistakes: {bnb_mistakes}")


    lr = LogisticRegression()
    lr.fit(xTrain, yTrain)
    lr_predictions = lr.predict(xTest)
    lr_accuracy = accuracy_score(yTest, lr_predictions)
    lr_mistakes = calc_mistakes(yTest, lr_predictions)
    print(f"Accuracy: {lr_accuracy:.2f}")
    print(f"Mistakes: {lr_mistakes}")


if __name__ == "__main__":
    main()
