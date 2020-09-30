
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import  CountVectorizer
import math
X = [[1,2,3], [3,4,5],[6,4,5] , [2,6,8] ]
y = [1,2,3]
y= np.array(y)
y[y>0]=0
print(y)








