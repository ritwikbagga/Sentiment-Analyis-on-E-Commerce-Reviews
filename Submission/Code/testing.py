
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import  CountVectorizer
def load_data():
    x_train = pd.read_csv("../../Data/X_train.csv")
    x_train = np.array(x_train["Review Text"])
    y_train = pd.read_csv("../../Data/y_train.csv")["Sentiment"]
    vectorizer = CountVectorizer()
    X= vectorizer.fit(x_train)
    X= vectorizer.transform(x_train)
    #print(vectorizer.get_feature_names())
    print(X.shape)










load_data()