"""
Utils for final project.
Authors:
Date:
"""

# python imports
from collections import OrderedDict
import math
import numpy as np
import optparse
import sys

# my file imports

def parse_args():
    """Parse command line arguments (train and test arff files)."""
    parser = optparse.OptionParser(description='run decision tree method')

    parser.add_option('-r', '--train_filename', type='string', help='path to' +\
        ' train arff file')
    parser.add_option('-e', '--test_filename', type='string', help='path to' +\
        ' test arff file')
    parser.add_option('-d', '--depth', type='int', help='max depth (optional)')

    (opts, args) = parser.parse_args()

    mandatories = ['train_filename', 'test_filename',]
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts

def read_csv(filename):
    """
    Read csv file into numpy array. Params:
    * filename (str), the path to the csv file
    """
    csv_file = open(filename,'r',encoding="utf8")
    examples = [] # matrix of examples
    labels = [] # list of labels
    names = []


    header = csv_file.readline()
    # read the examples and labels
    for line in csv_file:

        inQuote = False
        for index, char in enumerate(line):
            if char=="\"":
                inQuote = not inQuote
            if inQuote and char==",":
                line = line[:index]+" "+line[index+1:]

        tokens = line.split(",")

        label = tokens[1]
        label = int(label)
        example = tokens[2:]
        example = [float(feature) for feature in example]
        if example not in examples:
            labels.append(label)
            examples.append(example)
            names.append(tokens[0])

    csv_file.close()

    X = np.array(examples)
    y = np.array(labels)

    return X, y

if __name__ == "__main__":
    X,y = read_csv("19000-spotify-songs/song_data.csv")
    print(y.shape)
