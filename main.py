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

#Our files
import util

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
    X, y = util.read_csv(FILE)

    # Partition data into train and test datasets
    X_train, y_train, X_test, y_test = partition(X, y)
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    y_pred = testRandomForest(X_train, y_train, X_test, y_test, T, False)
    conf_mat(y_pred, X_test, y_test, "Random Forest")
    # print(testRandomForest(X_train, y_train, X_test, y_test, T, True))

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
