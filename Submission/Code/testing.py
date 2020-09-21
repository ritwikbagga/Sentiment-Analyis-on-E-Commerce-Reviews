
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import  CountVectorizer

y_train = pd.read_csv("../../Data/y_train.csv")
y_train.loc[y_train["Sentiment"] == 'Positive', "Sentiment"] = 1
y_train.loc[y_train["Sentiment"] == 'Negative', "Sentiment"] = 0
y_train = np.array(y_train["Sentiment"])

vectorizer = CountVectorizer(stop_words='english')
x_train = pd.read_csv("../../Data/X_train.csv")
x_train = x_train["Review Text"]
X = vectorizer.fit_transform(x_train)
X= X.toarray()
# print(X.shape)

positive_indices = np.argwhere(y_train == 1)
Negative_indices = np.argwhere(y_train == 0)
print(positive_indices[0])
X_p = X[positive_indices]
X_n = X[Negative_indices]
# print(X_p[0])
# print(X_p.shape)
# print(X_n.shape)




