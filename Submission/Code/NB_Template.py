import re
import numpy as np
import pandas as pd
import string
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


class BagOfWords(object):
    """
    Class for implementing Bag of Words
     for Q1.1
    """
    def __init__(self, vocabulary_size):
        """
        Initialize the BagOfWords model
        """
        self.vocabulary_size = vocabulary_size

    def preprocess(self, text):
        """
        Preprocessing of one Review Text
            - convert to lowercase done
            - remove punctuation
            - empty spaces
            - remove 1-letter words
            - split the sentence into words

        Return the split words
        """
        #lower case
        text = text.lower()
        words = text.split()
        puntuation = ""
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in words]
        final_words = []
        for word in stripped:    #remove one letter word
            if len(word)>1:
                final_words.append(word)

        return final_words




    def fit(self, X_train):
        """
        Building the vocabulary using X_train
        """
        pass
        
        
    def transform(self, X):
        """
        Transform the texts into word count vectors (representation matrix)
            using the fitted vocabulary
        """
        pass

class NaiveBayes(object):
    def __init__(self, beta=1, n_classes=2):
        """
        Initialize the Naive Bayes model
            w/ beta and n_classes
        """
        self.beta = beta
        self.n_classes = n_classes

    def fit(self, X_train, y_train):
        """
        Fit the model to X_train, y_train
            - build the conditional probabilities
            - and the prior probabilities
        """
        pass

    def predict(self, X_test):
        """
        Predict the X_test with the fitted model
        """
        pass


def confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix of the
        predictions with true labels
    """
    pass


def load_data(return_numpy=False):
    """
    Load data

    Params
    ------
    return_numpy:   when true return the representation of Review Text
                    using the CountVectorizer or BagOfWords
                    when false return the Review Text

    Return
    ------
    X_train
    y_train
    X_valid
    y_valid
    X_test
    """
    pass



def main():
    # Load in data
    X_train, y_train, X_valid, y_valid, X_test = load_data(return_numpy=False)
        
    # Fit the Bag of Words model for Q1.1
    bow = BagOfWords(vocabulary_size=10)
    bow.fit(X_train[:100])
    representation = bow.transform(X_train[100:200])

    # Load in data
    X_train, y_train, X_valid, y_valid, X_test = load_data(return_numpy=True)

    # Fit the Naive Bayes model for Q1.3
    nb = NaiveBayes(beta=1)
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_valid)
    print(confusion_matrix(y_valid, y_pred))


if __name__ == '__main__':
    main()
