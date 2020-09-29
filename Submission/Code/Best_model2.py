import math
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn import tree
import sklearn
from sklearn import naive_bayes
from sklearn import metrics
from sklearn.model_selection import KFold
from matplotlib.legend_handler import HandlerLine2D
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

def LR_classsifier(penalty, training_input, training_output, validation_input):
    print("###### LOGISTIC REGRESSION MODEL FOR  Q4 ######")
    print("Running...")
    clf = sklearn.linear_model.LogisticRegression(penalty=penalty, max_iter=1500)
    clf = clf.fit(training_input, training_output)
    y_prob = clf.predict_proba(validation_input)
    y_prob = y_prob[:, 1:]
    y_pred = clf.predict(validation_input)
    return y_pred , y_prob



def NB_classifier(training_input, training_output, validation_input , alpha):
    clf = naive_bayes.MultinomialNB(alpha=alpha)
    clf = clf.fit(training_input,training_output )
    y_pred = clf.predict(validation_input)
    y_prob = clf.predict_proba(validation_input)
    y_prob = y_prob[:, 1:]
    return y_pred , y_prob





def main():
    y_train = pd.read_csv("../../Data/Y_train.csv")
    y_train = (y_train['Sentiment'] == 'Positive').values.astype(int)
    y_train = np.array(y_train)
    y_valid = pd.read_csv("../../Data/Y_val.csv")
    y_valid = (y_valid['Sentiment'] == 'Positive').values.astype(int)
    y_valid = np.array(y_valid)
    vectorizer = CountVectorizer()
    x_train = pd.read_csv("../../Data/X_train.csv")
    x_train = x_train["Review Text"]
    x_valid = pd.read_csv("../../Data/X_val.csv")
    x_valid = x_valid["Review Text"]
    x_test = pd.read_csv("../../Data/X_test.csv")
    x_test = np.array(x_test["Review Text"])
    x_train = vectorizer.fit_transform(x_train)
    vocab = vectorizer.get_feature_names()
    vocab_size = len(vocab)
    x_train= x_train.toarray()
    x_valid = vectorizer.transform(x_valid)
    x_test = vectorizer.transform(x_test)
    x_test=x_test.toarray()




    ##### LOGISTIC REGRESSION ########
    y_pred , y_prob = LR_classsifier(penalty='l2', training_input= x_train, training_output= y_train, validation_input= x_valid)
    F1_score = f1_score(y_valid, y_pred)
    AUC = roc_auc_score(y_valid,y_prob )
    presicion = metrics.precision_score(y_valid, y_pred)
    recall = metrics.recall_score(y_valid,y_pred)
    print("Precision = " + str(presicion))
    print("Recall = " + str(recall))
    print("F1 SCORE =" + str(F1_score))
    print("AUC IS " + str(AUC))

    ##### Naive Bayes ########
    train_res = []
    test_res = []
    alpha_value = np.linspace(0,1,5)
    for alpha in alpha_value:

        y_pred, y_prob = NB_classifier( training_input=x_train, training_output=y_train, validation_input=x_valid , alpha = alpha)
        y_pred_train, y_prob_train = NB_classifier(training_input=x_train, training_output=y_train, validation_input=x_train, alpha=alpha)
        F1_score = f1_score(y_valid, y_pred)
        AUC = roc_auc_score(y_valid, y_prob)
        test_res.append(AUC)
        train_res.append(roc_auc_score(y_train, y_prob_train))
        presicion = metrics.precision_score(y_valid, y_pred)
        recall = metrics.recall_score(y_valid,y_pred)
        print("Alpha= "+str(alpha))
        print("BEST MODELPrecision = " + str(presicion))
        print("BEST MODEL Recall = " + str(recall))
        print("BEST MODEL F1 SCORE =" + str(F1_score) )
        print("BEST MODEL AUC IS " + str(AUC))

    plt.figure("Hyper-parameter tuning for alpha in Multinomial Naive Bayes ")
    line1, = plt.plot(alpha_value, train_res, 'b', label="Train AUC")
    line2, = plt.plot(alpha_value, test_res, 'r', label="Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("AUC SCORE")
    plt.xlabel("ALPHA VALUE")
    plt.show()


    #Naive Bayes is doing better so we use that for X_TEST with alpha =0.5 ##########
    y_pred, y_prob = NB_classifier(training_input=x_train, training_output=y_train, validation_input=x_test , alpha= 0.5)
    np.savetxt("Best_Predictions_x_test.csv",y_pred , delimiter=",")











if __name__ == '__main__':
    main()
