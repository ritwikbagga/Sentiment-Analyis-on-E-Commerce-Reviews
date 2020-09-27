import re
import math
import numpy as np
import pandas as pd
import string
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score , accuracy_score
from matplotlib import pyplot as plt


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
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in words]
        #to_remove =['and', 'but', 'in', 'is', 'it', 'of', 'the', 'this', 'to', 'with','that', 'for', 'my', 'have', 'as','was' , 'so' ]
        final_words = []
        for word in stripped:
            if len(word)>1: #remove one letter word
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
        x_t = X_train
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

        """
        vocab = sorted(self.vocabulary)  #vocab sorted
        vocab_index = {}
        for index,value in enumerate(vocab):
            vocab_index[value]=index
        # vector0 = np.zeros(self.vocabulary_size)  # [0,0,0,0,,,0]
        rep_matrix= []

        for text in X:
            words = self.preprocess(text) #list of words in one review text
            vector = np.zeros(self.vocabulary_size) #vector initially all zeros
            for word in words:
                #if word in vocab then increment index of vector[index of word in vocab]
                if word in vocab:
                    vector[vocab_index[word]]+=1
            rep_matrix.append(vector)

        # rep_matrix should be shape (100,10)
        matrix = np.array(rep_matrix)
        print(sorted(self.vocabulary))
        return matrix


class NaiveBayes(object):
    def __init__(self, beta=1, n_classes=2):
        """
        Initialize the Naive Bayes model
            w/ beta and n_classes
        """
        self.beta = beta
        self.n_classes = n_classes
        self.priors = {}
        self.vocab_dic = {}
        self.conditionals_p = dict()
        self.conditionals_n = dict()


    def fit(self, X_train, y_train, vocab):
        """

        """
        print("Fitting now")
        self.vocab= vocab
        for index, label in enumerate(y_train):
            if label not in self.priors:
                self.priors[label]=1
            else:
                self.priors[label]+=1
        #now we have count of total number of each label in prior we have to divide by |y_train|
        for label in np.unique(y_train):
            self.priors[label] = (self.priors[label]+(self.beta-1))/(len(y_train)+ (self.beta-1)*len(np.unique(y_train))) #priors are done

        print("priors are done")

        #lets build the conditionals
        positive_indices =  np.where(y_train==1)[0]
        Negative_indices = np.where(y_train == 0)[0]

        X_train_positives = X_train[positive_indices] #samples of label positives
        X_train_negatives = X_train[Negative_indices] #sample of label negative

        X_train_positive_counts = np.sum(X_train_positives,axis=0 ) #count of word label = pos

        X_train_negative_counts = np.sum(X_train_negatives, axis=0) #count of word label =neg

        for index_word, value in enumerate(vocab):
                 self.conditionals_p[index_word] = (X_train_positive_counts[index_word] + (self.beta -1))   / ( np.sum(X_train_positive_counts) + (self.beta-1)*2 )
                 self.conditionals_n[index_word] = (X_train_negative_counts[index_word] + (self.beta - 1))  / ( np.sum(X_train_negative_counts) + (self.beta-1)*2 )
        print("model is fitted")









    def predict(self, X_test):
        """
        Predict the X_test with the fitted model
        """
        print("predicting now")
        y_pred = []
        y_prob = []
        for x in X_test:
            positive_prob = math.log(self.priors[1])
            negative_prob = math.log(self.priors[0])
            for word_index, word in enumerate(x):
                positive_prob +=math.log( (self.conditionals_p[word_index])**word )  # "**word" because that is frequency
                negative_prob += math.log( (self.conditionals_n[word_index])**word ) # "**word" because that is frequency

            if positive_prob>negative_prob:
                y_pred.append(int(1))
            else:
                y_pred.append(int(0))

            y_prob.append(negative_prob/(negative_prob+positive_prob)) # probability of 1

        y_pred=np.array(y_pred)
        y_prob= np.array(y_prob)
        return y_pred , y_prob



def confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix of the
        predictions with true labels
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    confusion_matrix = np.zeros((2,2))

    for i in zip(y_true, y_pred):
        if i[0]==1:
            if i[1]==1:
                tp+=1
            else:
                fn+=1
        else: #y_true=0
            if i[1]==0:
                tn+=1
            else:
                fp+=1
    confusion_matrix[0][0] = tn
    confusion_matrix[0][1] = fp
    confusion_matrix[1][0] = fn
    confusion_matrix[1][1] = tp
    cm_df = pd.DataFrame(confusion_matrix)
    cm_df.columns = ['Predicted Negative', 'Predicted Positive']
    cm_df = cm_df.rename(index={0: 'Actual Negative', 1: 'Actual Positive'})
    return cm_df






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
    y_train = pd.read_csv("../../Data/Y_train.csv")
    y_train = (y_train['Sentiment'] == 'Positive').values.astype(int)

    y_train = np.array(y_train)
    y_valid = pd.read_csv("../../Data/Y_val.csv")
    y_valid = (y_valid['Sentiment'] == 'Positive').values.astype(int)
    y_valid = np.array(y_valid)






    if not return_numpy:
        x_train = pd.read_csv("../../Data/X_train.csv")
        x_train = np.array(x_train["Review Text"])
        x_valid = pd.read_csv("../../Data/X_val.csv")
        x_valid = np.array(x_valid["Review Text"])
        x_test = pd.read_csv("../../Data/X_test.csv")
        x_test = np.array(x_test["Review Text"])
        return x_train, y_train, x_valid, y_valid, x_test

    vectorizer = CountVectorizer()
    x_train = pd.read_csv("../../Data/X_train.csv")
    x_train = np.array(x_train["Review Text"])
    x_valid = pd.read_csv("../../Data/X_val.csv")
    x_valid = np.array(x_valid["Review Text"])
    x_test = pd.read_csv("../../Data/X_test.csv")
    x_test = np.array(x_test["Review Text"])
    x_train = vectorizer.fit_transform(x_train)
    vocab = vectorizer.get_feature_names()
    x_train= x_train.toarray()

    x_valid = vectorizer.transform(x_valid)
    x_valid= x_valid.toarray()

    x_test = vectorizer.transform(x_test)
    x_test=x_test.toarray()

    return x_train, y_train, x_valid, y_valid, x_test , vocab








def main():

    print("########## BAG of Words ##########")
    X_train, y_train, X_valid, y_valid, X_test = load_data(return_numpy=False)
    # Fit the Bag of Words model for Q1.1
    bow = BagOfWords(vocabulary_size=10)
    bow.fit(X_train[:100])
    representation = bow.transform(X_train[100:200])
    ret = np.sum(representation, axis=0)
    print(ret)


    print("########## Naive Bayes model N##########")
    # Load in data
    X_train, y_train, X_valid, y_valid, X_test , vocab = load_data(return_numpy=True)
    #
    # # #Fit the Naive Bayes model for Q1.3
    beta_list = [ 1.25, 1.5, 1.6,1.8]
    test_results=[]
    # train_results= []
    ROC_list = []
    for beta in beta_list:
        print("########## for Beta = "+ str(beta) + " ###############")
        nb = NaiveBayes(beta)
        nb.fit(X_train, y_train,vocab)
        y_pred , y_prob = nb.predict(X_valid)
        y_pred_train, y_prob_training = nb.predict(X_train) #used for tuning
        f1 = f1_score(y_valid, y_pred)
        accuracy = accuracy_score(y_valid,y_pred)
        R_Score_test = roc_auc_score(y_valid, y_prob)  #used for tuning
        ROC_score_train = roc_auc_score(y_train, y_prob_training) #used for tuning
        test_results.append(R_Score_test) #used for tuning
        # train_results.append(ROC_score_train) #used for tuning
        print("FINAL ROC AUC SCORE= " + str(R_Score_test) + " FOR Beta = " + str(beta))
        print("FINAL F1 SCORE= " + str(f1) + " FOR Beta = " + str(beta))
        print("Accuracy= " + str(accuracy) + " FOR Beta = " + str(beta))
        print(confusion_matrix(y_valid, y_pred))


    plt.figure("Hyper-parameter tuning for beta")
    # line1, = plt.plot(beta_list, train_results, 'b', label ="Train AUC")
    line2 = plt.plot(beta_list, test_results, 'r', label ="Test AUC")
    plt.ylabel("AUC SCORE")
    plt.xlabel("VALUE OF BETA")
    plt.show() #we can see how how we start to overfit with increasing depth.  we can choose d=6 for our  best model



if __name__ == '__main__':
    main()
