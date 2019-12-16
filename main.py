"""
This file provides methods to run a dataset through the pipeline for our project.
Authors: Brian Becker and Hannah Beilinson
Date: 12/4/2019
"""

import optparse
import sys

# Package imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns

import tensorflow as tf

#Our files
import util
import run_nn_tf as nn
from fc_nn import FCmodel

FILE = "19000-spotify-songs/song_data.csv"
T = 200

def partition(X, y):
    # Partitioned the same way each time it runs so that we're not cross contaminating
    n, p = X.shape
    train_size = int(n*0.8)

    train_X = X[:train_size]
    train_y = y[:train_size]
    test_X = X[train_size:]
    test_y = y[train_size:]

    return train_X, train_y, test_X, test_y

def main():
    # Runs pipeline

    # Read in data from csv
    X, y = util.read_csv(FILE,normalize=True,mean_center=True)

    # Partition data into train and test datasets
    X_train, y_train, X_test, y_test = partition(X, y)
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    print(testRandomForest(X_train, y_train, X_test, y_test, T, True))
    run_fc_nn(X_train,y_train,X_test,y_test)

def trainRandomForest(X,y,T,regressor=False):
    """Trains Random Forest on train sets X and y"""
    if regressor:
        clf = RandomForestRegressor(n_estimators=T)
    else:
        clf = RandomForestClassifier(n_estimators=T,criterion="entropy")

    clf.fit(X,y)
    return clf

def testRandomForest(X_train,y_train,X_test,y_test,T,regressor=False):

    clf = trainRandomForest(X_train,y_train,T,regressor)
    yHat = np.array(clf.predict(X_test))
    return yHat

def run_fc_nn(X_train, y_train, X_test, y_test):
    # set up train_dset, val_dset, and test_dset:
    # see documentation for tf.data.Dataset.from_tensor_slices, use batch = 64
    # train should be shuffled, but not validation and testing datasets
    train_dset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
    test_dset = tf.data.Dataset.from_tensor_slices((X_test,y_test))

    for songs,labels in train_dset:
        print(songs.shape)
        print(labels.shape)

    ######## END YOUR CODE #############

    ###### TODO: YOUR CODE HERE ######
    # call the train function to train a fully connected neural network
    fc = FCmodel()
    train_acc,epochs = nn.run_training(fc,train_dse)
    print(train_acc)
    #train_curve(train_acc,val_acc,epochs,"FCcurve.png")

    ######## END YOUR CODE #############

def accuracy(y,yHat):
    """Returns accuracy of predictions yHat on true labels y in discrete case."""
    correct = [(yHat[i]==y[i]) for i in range(len(yHat))]
    return np.sum(correct)/len(correct)

def rMSE(y,yHat):
    n = len(y)
    difs = yHat - y
    sDifs = [e**2 for e in dif]
    mse = np.sum(sDifs)/n
    return math.sqrt(mse)

def conf_mat(y_pred, X_test, y_test, regressor_name):
    matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(matrix,annot=False,cbar=False)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(regressor_name + ' Confusion Matrix')
    plt.show()
    print(matrix)

if __name__=="__main__":
    main()
