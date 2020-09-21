
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import  CountVectorizer

vectorizer = CountVectorizer()
x_train = pd.read_csv("../../Data/X_train.csv")
x_train = np.array(x_train["Review Text"])
vectorizer.fit(x_train)
x_train = vectorizer.transform(x_train)
x_train.toarray()

y_train = pd.read_csv("../../Data/y_train.csv")
y_train.loc[y_train["Sentiment"] == 'Positive', "Sentiment"] = 1
y_train.loc[y_train["Sentiment"] == 'Negative', "Sentiment"] = 0
y_train = np.array(y_train["Sentiment"])

positive_indices = np.where(y_train == 1)[0]
Negative_indices = np.where(y_train == 0)[0]
X_train_positives = x_train[positive_indices]
X_train_negatives = x_train[Negative_indices]
X_train_positives_conditionals = np.sum( X_train_positives,  axis=0)
X_train_negative_conditionals = np.sum(X_train_negatives, axis=0).reshape(-1)
X_train_positives_conditionals= np.array(X_train_positives_conditionals)
print(X_train_positives_conditionals)
# for word , index in enumerate(X_train_positives_conditionals):
#     if index == 5:
#         break
#     print(word)



