import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing


def model_assessment(filename):
    """
    Given the entire data, split it into training and test set 
    so you can assess your different models to compare perceptron, logistic regression, and naive bayes. 
    """

    with open(filename, 'r') as file:
        data = file.read().splitlines()

    yTotal = [int(line.split()[0]) for line in data]
    xTotal = [line[1:] for line in data]

    df = pd.DataFrame({'y': yTotal, 'text': xTotal})
    
    sample_size = int(0.8 * len(df))
    selected_indices = np.random.choice(len(df), sample_size, replace=False)
    Train = df.loc[selected_indices]
    Test = df.drop(selected_indices)

    return Train, Test

def build_vocab_map(traindf):
    """
    Construct the vocabulary map such that it returns
    (1) the vocabulary dictionary contains words as keys and
    the number of emails the word appears in as values, and
    (2) a list of words that appear in at least 30 emails.

    ---input:
    dataset: pandas dataframe containing the 'text' column
             and 'y' label column

    ---output:
    dict: key-value is word-count pair
    list: list of words that appear in at least 30 emails
    """
    vocab = {}
    word_list = []

    for y, text in traindf.iterrows(): 
        row_vocab = set()
        for word in text:
            row_vocab.add(word)

        for word in row_vocab:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
    for word in vocab:
        if vocab[word] >= 30:
            word_list.append(word)

    return vocab, word_list


def construct_binary(dataset, freq_words):
    """
    Construct email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise

    ---input:
    dataset: pandas dataframe containing the 'text' column

    freq_word: the vocabulary map built in build_vocab_map()

    ---output:
    numpy array
    """
    return None


def construct_count(dataset, freq_words):
    """
    Construct email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise

    ---input:
    dataset: pandas dataframe containing the 'text' column

    freq_word: the vocabulary map built in build_vocab_map()

    ---output:
    numpy array
    """
    return None


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()
    Train, Test = model_assessment(args.data)
    build_vocab_map(Train)
    construct_binary()
    construct_count()



if __name__ == "__main__":
    main()
