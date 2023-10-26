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
    xTotal = [''.join(line[1:]) for line in data]


    df = pd.DataFrame({'y': yTotal, 'text': xTotal})

    df['text'] = df['text'].apply(lambda text: text.split()[0:])

    
    sample_size = int(0.8 * len(df))
    selected_indices = np.random.choice(len(df), sample_size, replace=False)
    Train = df.loc[selected_indices]
    Test = df.drop(selected_indices)

    return Train, Test

def build_vocab_map(traindf):
    vocab = {}
    word_list = []

    for text_list in traindf['text']:

        email_words = set()

        for word in text_list:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1

            email_words.add(word)

        for word in email_words:
            if vocab[word] >= 30 and word not in word_list:
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

    complete_array = []
    for text_list in dataset['text']:

        email_words = set(text_list)
        
        binary_list = []

        for vocab in freq_words:
            if vocab in email_words:
                binary_list.append(1)
            else:
                binary_list.append(0)
        complete_array.append(binary_list)
        
    return np.array(complete_array)


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
    complete_array = []
    for text in dataset['text']:

        if isinstance(text, list):
            text = ' '.join(text)
        else:
            text = text 

        email_word_count = text.split()
        count_list = [email_word_count.count(word) for word in freq_words]

        complete_array.append(count_list)

    return np.array(complete_array)


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
    vocab_dict, word_30 = build_vocab_map(Train)
    construct_binary(Train, word_30)
    construct_count(Train, word_30)



if __name__ == "__main__":
    main()
