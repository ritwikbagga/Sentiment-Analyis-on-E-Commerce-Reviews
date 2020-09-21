
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import  CountVectorizer

vectorizer = CountVectorizer(stop_words="english")
x_train = pd.read_csv("../../Data/X_train.csv")
x_train = np.array(x_train["Review Text"])
vectorizer.fit(x_train)
x_train = vectorizer.transform(x_train)
x_train.toarray()

vocab = {}
v = vectorizer.get_feature_names()

for index, value in enumerate(v):
    vocab[value] = index

y_train = pd.read_csv("../../Data/y_train.csv")
y_train.loc[y_train["Sentiment"] == 'Positive', "Sentiment"] = 1
y_train.loc[y_train["Sentiment"] == 'Negative', "Sentiment"] = 0
y_train = np.array(y_train["Sentiment"])

positive_indices = np.where(y_train == 1)[0]
Negative_indices = np.where(y_train == 0)[0]
X_train_positives = x_train[positive_indices]
X_train_negatives = x_train[Negative_indices]

print(vocab)




