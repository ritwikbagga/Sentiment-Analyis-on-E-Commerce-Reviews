
import numpy as np
import pandas as pd
def load_data():
    # x_train = pd.read_csv("../../Data/X_train.csv")
    # x_train = np.array(x_train["Review Text"])
    y_train = pd.read_csv("../../Data/y_train.csv")["Sentiment"]
    print(y_train.shape)




load_data()