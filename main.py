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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import math
#import seaborn as sns

#import tensorflow as tf

#Our files
import util
#import run_nn_tf as nn
#from fc_nn import FCmodel

FILE = "19000-spotify-songs/song_data.csv"
T = 1000

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
    accuracy_vs_binning(False)

    # Read in data from csv
    # X, y = util.read_csv(FILE,normalize=True,mean_center=True, do_bin=True, bin_step=20)


    #test correlation function
    testCor(X,y)

    # Partition data into train and test datasets
    # X_train, y_train, X_test, y_test = partition(X, y)

    # Uncomment below to test Random Forest
    # run_pipeline_rf(X_train, y_train, X_test, y_test)

    #Uncomment below to test sklearn FC
    # run_pipeline_mlp(X_train, y_train, X_test, y_test)

    # Uncomment below to test tensorflow FC
    # run_fc_nn(X_train,y_train,X_test,y_test)

def run_pipeline_rf(X_train, y_train, X_test, y_test):
    y_pred = testRandomForest(X_train,y_train,X_test,y_test,T,regressor=False)
    # print(rMSE(y_test, y_pred))
    print(accuracy(y_test, y_pred))
    conf_mat(y_pred, X_test, y_test, "Random Forest", numbers=False)

def run_pipeline_mlp(X_train, y_train, X_test, y_test):
    y_pred = testMLP(X_train,y_train,X_test,y_test)
    # print(rMSE(y_test, y_pred))
    print(accuracy(y_test, y_pred))
    conf_mat(y_pred, X_test, y_test, "MLP", numbers=False)

def trainMLP(X,y):
    clf = MLPClassifier(max_iter=1000)
    clf.fit(X,y)
    return clf

def testMLP(X_train,y_train,X_test,y_test):

    clf = trainMLP(X_train,y_train)
    yHat = np.array(clf.predict(X_test))
    return yHat

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

# def run_fc_nn(X_train, y_train, X_test, y_test):
#     # set up train_dset, val_dset, and test_dset:
#     # see documentation for tf.data.Dataset.from_tensor_slices, use batch = 64
#     # train should be shuffled, but not validation and testing datasets
#     train_dset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
#     test_dset = tf.data.Dataset.from_tensor_slices((X_test,y_test))

#     for songs,labels in train_dset:
#         print(songs.shape)
#         print(labels.shape)

#     ######## END YOUR CODE #############

#     ###### TODO: YOUR CODE HERE ######
#     # call the train function to train a fully connected neural network
#     fc = FCmodel()
#     train_acc,epochs = nn.run_training(fc,train_dset)
#     print(train_acc)
#     #train_curve(train_acc,val_acc,epochs,"FCcurve.png")

#     ######## END YOUR CODE #############

def accuracy(y,yHat):
    """Returns accuracy of predictions yHat on true labels y in discrete case."""
    correct = [(yHat[i]==y[i]) for i in range(len(yHat))]
    return np.sum(correct)/len(correct)

def rMSE(y,yHat):
    n = len(y)
    difs = yHat - y
    sDifs = [e**2 for e in difs]
    mse = np.sum(sDifs)/n
    return math.sqrt(mse)

def conf_mat(y_pred, X_test, y_test, regressor_name, numbers=False):
    matrix = confusion_matrix(y_test, y_pred)
    #sns.heatmap(matrix,annot=numbers,cbar=False)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(regressor_name + ' Confusion Matrix')
    plt.show()
    print(matrix)


def correlation_plot():
    pass

def accuracy_vs_binning(proportional):
    bin_steps = [1, 5, 10, 20, 25, 50, 100]
    RF_accuracies = []
    MLP_accuracies = []
    for step_size in bin_steps:
        X, y = util.read_csv(FILE,normalize=True,mean_center=True, do_bin=True, bin_step=step_size)
        X_train, y_train, X_test, y_test = partition(X, y)
        # Test with random forest
        rf_y_hat = testRandomForest(X_train,y_train,X_test,y_test,T,regressor=False)
        RF_accuracies.append(accuracy(y_test, rf_y_hat))
        # Test with MLP
        mlp_y_hat = testMLP(X_train,y_train,X_test,y_test)
        MLP_accuracies.append(accuracy(y_test, mlp_y_hat))

    plt.plot(RF_accuracies, bin_steps)
    plt.plot(MLP_accuracies, bin_steps)
    plt.legend(['Random Forest', "Neural Network"])
    plt.ylabel("Accuracy")
    plt.xlabel("Binning Step Size")
    plt.title("Accuracy vs. Bin Size")
    plt.show()
    plt.savefig("Accuracy vs. Binning.png")


def testCor(X,y):
    """Tests correlation given ftr mtx and label vec"""
    for i in range(len(X[0])):
        col = X[:,i]
        print("FEATURE %d"%(i))
        correlation(y,col)


def correlation(y,x):
    """Computes sample correlation between observed vectors
    y and x"""
    z = np.stack((x,y),axis=0)
    cor = np.corrcoef(z)
    print(cor)
    return cor[0][1]


if __name__=="__main__":
    main()
