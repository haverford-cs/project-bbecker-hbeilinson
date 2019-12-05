"""
This file provides methods to run a dataset through the pipeline for our project.
Authors: Brian Becker and Hannah Beilinson
Date: 12/4/2019
"""

import optparse
import sys

# Package imports
from sklearn.ensemble import RandomForestClassifier
import numpy as np

#Our files
import util

FILE = "19000-spotify-songs/song_data.csv"

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

    # X = np.array([
    # [1,1,1,1],
    # [2,2,2,2],
    # [3,3,3,3],
    # [4,4,4,4],
    # [5,5,5,5],
    # ])
    # y = [1,2,3,4,5]

    # Partition data into train and test datasets
    train_X, train_y, test_X, test_y = partition(X, y)
    print (train_X, train_y, test_X, test_y)


if __name__=="__main__":
    main()
