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
    cm_df = pd.DataFrame(confusion_matrix)
    cm_df.columns = ['Predicted Negative', 'Predicted Positive']
    cm_df = cm_df.rename(index={0: 'Actual Negative', 1: 'Actual Positive'})
    return cm_df

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
    return sig

  def hx(self, X_train):
        y_hat = self.sigmoid(self.W * X_train.T)
        return y_hat

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
    loss = (-1/N)*np.sum(  y*np.log(y_hat) + (1-y)*np.log(1-y_hat))+ self.reg*np.sum(self.reg**2)
    #TODO: Compute gradients

    # Calculate dLdW meaning the gradient of loss function according to W 
    # you can use chain rule here with calculating each step to make your job easier
    dLdW = -(1/N)*(X.T*( y_hat-y) ) + 2*self.reg*self.W
    
    return loss, dLdW

  def gradDescent(self,X, y, learning_rate, num_epochs):

    N, D = X.shape
    loss_hist = np.zeros(num_epochs)
    for i in range(num_epochs):
      #TODO: implement steps of gradient descent
      loss, dLdW = self.loss(X, y)

      self.W = self.W + learning_rate*dLdW
      # printing loss, you can also print accuracy here after few iterations to see how your model is doing
      #print("Epoch : ", i, " loss : ", loss)
      
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
    #y_train = np.array(y_train)
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
   #x_train= x_train.toarray()
    x_valid = vectorizer.transform(x_valid)
    x_test = vectorizer.transform(x_test)
    x_test=x_test.toarray()
    #regularization weight  0 to 0.2
    #number of interations
    #learning rate




    # ############## For optimizing regularization weight ########
    # regularization_weights = np.linspace(0,0.05, num=5)
    # ROC_reg = []
    # for weight in regularization_weights:
    #     lr = LogisticRegression(input_size=vocab_size , reg=weight)
    #     #X, y, learning_rate, num_epochs
    #     losshis = lr.gradDescent(x_train, y_train, learning_rate=0.5, num_epochs=1000)
    #     y_pred , y_prob = lr.predict(x_valid)
    #     auc = getauc(y_valid, y_prob)
    #     ROC_reg.append(auc)
    #     print("AUC score for current model is "+str(auc) + " and this is for optimizing reg = "+ str(weight))
    #
    # plt.figure("ROC FOR DIFFERENT VALUES OF regularization_weights ")
    # plt.plot(regularization_weights, ROC_reg)
    # plt.xlabel("Value of regularization_weights")
    # plt.ylabel("ROC score")
    # plt.show()
    #
    #
    #  ############# For optimizing Learning rate ########
    # learning_rates = np.linspace(10**-4, 10, 7)
    # ROC_lr = []
    # for learning_rate in learning_rates:
    #     lr = LogisticRegression(input_size=vocab_size, reg=0.0125)
    #     #X, y, learning_rate, num_epochs
    #     losshis = lr.gradDescent(x_train, y_train, learning_rate=learning_rate, num_epochs=100)
    #     y_pred , y_prob = lr.predict(x_valid)
    #     auc = getauc(y_valid, y_prob)
    #     ROC_lr.append(auc)
    #     print("AUC score for current model is "+str(auc) + " and this is for optimizing learning rate which is now = "+ str(learning_rate))
    # plt.figure("ROC FOR DIFFERENT VALUES OF Learning rate ")
    # plt.plot(learning_rates, ROC_lr)
    # plt.xlabel("Value of Learning Rate")
    # plt.ylabel("ROC score")
    # plt.show()
    #
    #
    #
    # # ############## For optimizing no. of iterations rate ########
    # num_iterations = [100,500,1000,1500]
    # ROC_ni = []
    # for epochs in num_iterations:
    #     lr = LogisticRegression(input_size=vocab_size, reg=0.0125)
    #     losshis = lr.gradDescent(x_train, y_train, learning_rate=0.77, num_epochs=epochs)
    #     y_pred , y_prob = lr.predict(x_valid)
    #     auc = getauc(y_valid, y_prob)
    #     ROC_ni.append(auc)
    #     print("AUC score for current model is "+str(auc) + " and this is for optimizing number of iterations which is now = "+ str(epochs))
    # plt.figure("ROC FOR DIFFERENT VALUES OF iterations ")
    # plt.plot(num_iterations, ROC_ni)
    # plt.xlabel("Value of iterations")
    # plt.ylabel("ROC score")
    # plt.show()
    # best reg = 0.125
    #best learning rate = 0.8
    #best number of iterations = 1500

    ##################### Q1  ################
    lr_2 = LogisticRegression(input_size=vocab_size, reg = 0.0125)
    loss_2 = lr_2.gradDescent(x_train, y_train, learning_rate=0.8, num_epochs=1500  )
    y_pred_train , y_prob_train = lr_2.predict(x_train)
    auc_train = getauc(y_train, y_prob_train)
    y_pred_test , y_prob_test = lr_2.predict(x_valid)
    auc_test = getauc(y_valid, y_prob_test)
    #("train set auc= " + str(auc_train))
    print("test set auc" + str(auc_test))
    print(conf_mat(y_valid, y_pred_test)) ###### Q3 ########









    
    #TODO: Preprocess the data, here we will only select Review Text column in both train and validation and use CountVectorizer from sklearn to get bag of word representation of the review texts
    # Careful that you should fit vectorizer only on train data and use the same vectorizer for transforming X_train and X_val 
    
    # Write a for loop for each hyper parameter here each time initialize logistic regression train it on the train data and get auc on validation data and get confusion matrix using the best hyper params 
    
if __name__ == '__main__':
    main()



