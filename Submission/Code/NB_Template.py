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
        self.vocabulary=[]

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
        #x_train is already preprocessed?
        #for each review in X_train it is a text then call the preprocess method only first 100
        #to get the split words
        #create a dictionary and for first 100 reviews get the count
        #create a vocab of most frequent 10 words in first 100 samples
        """
        vocab = {}
        x_t = X_train[0:100] #x_t has first 100 texts
        for text in x_t:
            split_text = self.preprocess(text) #got the list of words
            for word in split_text:
                if word not in vocab:
                    vocab[word]=1
                else:
                    vocab[word]+=1

        #now we have the dictionary , we will sort it by freuency desc
        sort_Vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        cur = 0
        for w in sort_Vocab:
            if cur==self.vocabulary_size:
                break
            self.vocabulary.append(w[0])
            cur+=1

        #we now have model vocabulary of size vocab.size


    def transform(self, X):
        """
Transform the texts into word count vectors (representation matrix)using the fitted vocabulary
then transform the next 100 entries into a 100 by 10 representation matrix X = {X1, X2, ..., X100}.
For this question, you need to report the total word count of the vocabulary V in the 100 by 10
representation matrix, and the word count of the V in the representation matrix X should be a vector
of size 10. (hint: sum the representation matrix along the first axis)
        """
        vocab = sorted(self.vocabulary)  #vocab sorted
        vocab_index = {}
        for index,value in enumerate(vocab):
            vocab_index[value]=index
        rep_matrix=[]
        vector0 = np.zeros(self.vocabulary_size)  #[0,0,0,0,,,0]
        X = X.T #now we have text1, text2,text3...text100

        for text in X:
            words = self.preprocess(text) #list of words in one review text
            vector = vector0 #vector initially all zeros
            for word in words:
                #if word in vocab then increment index of vector[index of word in vocab]
                if word in vocab:
                    vector[vocab_index[word]]+=1


            rep_matrix.append(vector)
        # rep_matrix should be shape (100,10)
        rep_matrix=np.array(rep_matrix)
        return rep_matrix









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
    if not return_numpy:
        x_train = pd.read_csv("../../Data/X_train.csv")
        x_train = np.array(x_train["Review Text"])
        y_train = pd.read_csv("../../Data/y_train.csv")
        y_train = np.array(y_train["Sentiment"])
        x_valid = pd.read_csv("../../Data/X_val.csv")
        x_valid = np.array(x_valid["Review Text"])
        y_valid = pd.read_csv("../../Data/Y_val.csv.csv")
        y_valid = np.array(y_valid["Sentiment"])
        x_test = pd.read_csv("../../Data/X_test.csv")
        x_test = np.array(x_test["Review Text"])

        return x_train , y_train , x_valid , y_valid , x_test






def main():
    # Load in data
    X_train, y_train, X_valid, y_valid, X_test = load_data(return_numpy=False)
        
    # Fit the Bag of Words model for Q1.1
    bow = BagOfWords(vocabulary_size=10)
    bow.fit(X_train[:100])
    representation = bow.transform(X_train[100:200])
    ret = np.sum(representation, axis=0)
    print(ret.shape)

    # Load in data
    X_train, y_train, X_valid, y_valid, X_test = load_data(return_numpy=True)

    # Fit the Naive Bayes model for Q1.3
    nb = NaiveBayes(beta=1)
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_valid)
    print(confusion_matrix(y_valid, y_pred))


if __name__ == '__main__':
    main()
