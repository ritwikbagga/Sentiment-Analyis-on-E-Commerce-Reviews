
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import  CountVectorizer
def load_data(return_numpy=False):
    y_train = pd.read_csv("../../Data/y_train.csv")
    y_train.loc[y_train["Sentiment"] == 'Positive', "Sentiment"] = 1
    y_train.loc[y_train["Sentiment"] == 'Negative', "Sentiment"] = 0
    y_train = np.array(y_train["Sentiment"])

    y_valid = pd.read_csv("../../Data/Y_val.csv")
    y_valid.loc[y_valid["Sentiment"] == 'Positive', "Sentiment"] == 1
    y_valid.loc[y_valid["Sentiment"] == 'Negative', "Sentiment"] == 0
    y_valid = np.array(y_valid["Sentiment"])
    if not return_numpy:
        x_train = pd.read_csv("../../Data/X_train.csv")
        x_train = np.array(x_train["Review Text"])
        x_valid = pd.read_csv("../../Data/X_val.csv")
        x_valid = np.array(x_valid["Review Text"])
        x_test = pd.read_csv("../../Data/X_test.csv")
        x_test = np.array(x_test["Review Text"])
    else:
        vectorizer = CountVectorizer()
        x_train = pd.read_csv("../../Data/X_train.csv")
        x_train = np.array(x_train["Review Text"])
        x_valid = pd.read_csv("../../Data/X_val.csv")
        x_valid = np.array(x_valid["Review Text"])
        x_test = pd.read_csv("../../Data/X_test.csv")
        x_test = np.array(x_test["Review Text"])
        vectorizer.fit(x_train)

        x_train = vectorizer.transform(x_train)
        x_valid = vectorizer.transform(x_valid)
        x_test = vectorizer.transform(x_test)


    return x_train , y_train , x_valid , y_valid , x_test

x_train, y_train, X_valid, y_valid, X_test = load_data(return_numpy=True)
print(x_train.shape)
print(y_train.shape)
print(X_valid.shape)
print( y_valid.shape )
print(X_test.shape)













load_data()