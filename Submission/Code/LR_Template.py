from __future__ import print_function
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score

def getauc(y_true, probs):
    """
    Use sklearn roc_auc_score to get the auc given predicted probabilities and the true labels
    
    Args:
        - y_true: The true labels for the data
        - probs: predicted probabilities
    """
    #TODO: return auc using sklearn roc_auc_score
    scores = roc_auc_score(y_true, probs)
    return scores

def conf_mat(y_true, y_pred):
    """
    The method for calculating confusion matrix, you have to implement this by yourself.
    
    Args:
        - y_true: the true labels for the data
        - y_pred: the predicted labels
    """
    #TODO: compute and return confusion matrix
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    confusion_matrix = np.zeros((2, 2))

    for i in zip(y_true, y_pred):
        if i[0] == 1:
            if i[1] == 1:
                tp += 1
            else:
                fn += 1
        else:  # y_true=0
            if i[1] == 0:
                tn += 1
            else:
                fp += 1
    confusion_matrix[0][0] = tn
    confusion_matrix[0][1] = fp
    confusion_matrix[1][0] = fn
    confusion_matrix[1][1] = tp
    return confusion_matrix

    
class LogisticRegression(object):
  def __init__(self, input_size, reg=0.0, std=1e-4):
    """
    Initializing the weight vector
    
    Input:
    - input_size: the number of features in dataset, for bag of words it would be number of unique words in your training data
    - reg: the l2 regularization weight
    - std: the variance of initialized weights
    """
    self.W = std * np.random.randn(input_size)
    self.reg = reg

    
  def sigmoid(self,x):
    """
    Compute sigmoid of x
    
    Input:
    - x: Input data of shape (N,)
    
    Returns:
    - sigmoid of each value in x with the same shape as x (N,)
    """
    sig =  1/(1+np.exp(-x))
    sig[sig >= 0.999999] = 0.999999
    sig[sig <= 0.000001] = 0.000001
    sig = np.array(sig)
    return sig

  def hx(self, X_train):
    # shape of x_train = (N,D)
    #shape of w = (D,)
    z = np.dot(X_train, self.W.T)
    h = self.sigmoid(z)
    h=np.array(h)
    return h




  def loss(self, X, y):
    """
    Compute the loss and gradients for your logistic regression.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: A numpy array f shape (N,) giving training labels.

    Returns:
    - loss: Loss (data loss and regularization loss) for the entire training samples
    - dLdW: gradient of loss with respect to W
    """
    N, D = X.shape
    reg = self.reg
    
    #TODO: Compute scores
    y_hat = self.hx(X) #now we have y_hat of shape (len(X) , ) these are probabilities
    #TODO: Compute the loss
    loss = 1/N - np.sum(  y*np.log(y_hat) + (1-y)*np.log(1-y_hat) )
    #TODO: Compute gradients

    # Calculate dLdW meaning the gradient of loss function according to W 
    # you can use chain rule here with calculating each step to make your job easier
    dLdW = 1/N*np.dot(X.T, y_hat-y)
    
    return loss, dLdW

  def gradDescent(self,X, y, learning_rate, num_epochs):
    """
    We will use Gradient Descent for updating our weights, here we used gradient descent instead of stochastic gradient descent for easier implementation
    so you will use the whole training set for each update of weights instead of using batches of data.
    
    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - num_epochs: integer giving the number of epochs to train
    
    Returns:
    - loss_hist: numpy array of size (num_epochs,) giving the loss value over epochs
    """
    N, D = X.shape
    loss_hist = np.zeros(num_epochs)
    for i in range(num_epochs):
      #TODO: implement steps of gradient descent
      loss, dLdW = self.loss(X, y)
      self.W = self.W - learning_rate*dLdW
      # printing loss, you can also print accuracy here after few iterations to see how your model is doing
      print("Epoch : ", i, " loss : ", loss)
      
    return loss_hist

  def predict(self, X):
    """
    Use the trained weights to predict labels for data given as X

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
        - probs: A numpy array of shape (N,) giving predicted probabilities for each of the elements of X.
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of the elements of X. You can get this by putting a threshold of 0.5 on probs
    """
    #TODO: get the scores (probabilities) for input data X and calculate the labels (0 and 1 for each input data) and return them
    y_probs = self.hx(X)
    y_pred  = y_probs
    y_pred[y_pred>=0.5]=1
    y_pred[y_pred<0.5]=0
    return y_probs, y_pred

def main():
    y_train = pd.read_csv("../../Data/Y_train.csv")
    y_train = (y_train['Sentiment'] == 'Positive').values.astype(int)
    y_train = np.array(y_train)
    y_valid = pd.read_csv("../../Data/Y_val.csv")
    y_valid = (y_valid['Sentiment'] == 'Positive').values.astype(int)
    y_valid = np.array(y_valid)
    vectorizer = CountVectorizer()
    x_train = pd.read_csv("../../Data/X_train.csv")
    x_train = np.array(x_train["Review Text"])
    x_valid = pd.read_csv("../../Data/X_val.csv")
    x_valid = np.array(x_valid["Review Text"])
    x_test = pd.read_csv("../../Data/X_test.csv")
    x_test = np.array(x_test["Review Text"])
    x_train = vectorizer.fit_transform(x_train)
    vocab = vectorizer.get_feature_names()
    vocab_size = len(vocab)
    x_train= x_train.toarray()
    x_valid = vectorizer.transform(x_valid)
    x_valid= x_valid.toarray()
    x_test = vectorizer.transform(x_test)
    x_test=x_test.toarray()
    #regularization weight  0 to 0.2
    #number of interations
    #learning rate
    regularization_weights = [0.1]
    number_of_interations = [10]
    learning_rates = [10**-4]
    for weight in regularization_weights:
        lr = LogisticRegression(input_size=vocab_size)
        #X, y, learning_rate, num_epochs
        losshis = lr.gradDescent(x_train, y_train, learning_rate=10**-2, num_epochs=100)
        y_pred , y_prob = lr.predict(x_valid)
        auc = getauc(y_valid, y_prob)
        print("AUC score for current model is "+str(auc))




    
    #TODO: Preprocess the data, here we will only select Review Text column in both train and validation and use CountVectorizer from sklearn to get bag of word representation of the review texts
    # Careful that you should fit vectorizer only on train data and use the same vectorizer for transforming X_train and X_val 
    
    # Write a for loop for each hyper parameter here each time initialize logistic regression train it on the train data and get auc on validation data and get confusion matrix using the best hyper params 
    
if __name__ == '__main__':
    main()



