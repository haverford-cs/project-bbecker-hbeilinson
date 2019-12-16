"""
Starter code for NN training and testing.
Source: Stanford CS231n course materials, modified by Sara Mathieson
Authors:
Date:
"""

import matplotlib.pyplot as plt
import numpy as np
import os
# import tensorflow as tf

# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.datasets.cifar import load_batch

from fc_nn import FCmodel
##################





# @tf.function
def our_train_step(model,lossFunc,optimizer,x,y): # TODO what arguments?
    ###### TODO: YOUR CODE HERE ######
    # look up documentation for tf.GradientTape
    # compute the predictions given the images, then compute the loss
    # compute the gradient with respect to the model parameters (weights), then
    # apply this gradient to update the weights (i.e. gradient descent)

    with tf.GradientTape() as tape:
        predictions = model.call(x)
        print(predictions)
        loss = lossFunc(y,predictions)
    # Simultaneously optimize trunk and head1 weights.
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    ######## END YOUR CODE #############
    # return the loss and predictions
    return loss, predictions

@tf.function
def val_step(model,lossFunc,optimizer,x,y): # TODO what arguments?
    ###### TODO: YOUR CODE HERE ######
    # compute the predictions given the images, then compute the loss
    with tf.GradientTape() as tape:
        predictions = model.call(x)
        loss = lossFunc(y,predictions)
    ######## END YOUR CODE #############

    # return the loss and predictions
    return loss, predictions

def confusion_matrix(model,x,y,confMat):
    predictions = model.call(x)
    npLab = np.array(y)
    #print(npLab)
    npPred = np.array(predictions)
    #print(npPred)
    for i in range(10):
        for j in range(10):
            cell = [(npLab[k]==i and np.argmax(predictions[k])==j) for k in range(len(y))]
            cell = np.sum(cell)
            confMat[i][j] += cell

    return confMat


def run_training(model, train_dset):

    ###### TODO: YOUR CODE HERE ######
    # set up a loss_object (sparse categorical cross entropy)
    # use the Adam optimizer

    cce = tf.keras.losses.SparseCategoricalCrossentropy()
    adam = tf.keras.optimizers.Adam()

    ######## END YOUR CODE #############

    # set up metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy( \
        name='train_accuracy')

    ###### TODO: YOUR CODE HERE ######
    # train for 10 epochs (passes over the data)
    # Example of iterating over the data once:

    epochs = range(10)

    train_acc = []

    for epoch in epochs:
        for songs, labels in train_dset:
        # TODO run training step
            # print(songs, labels)
            loss, predictions = our_train_step(model,cce,adam,songs,labels)
        # uncomment below
            train_loss(loss)
            train_accuracy(labels, predictions)


        template = 'Epoch {}, Loss: {}, Accuracy: {}'
        ta = train_accuracy.result()
        train_acc.append(ta)

        print(template.format(epoch+1,
                            train_loss.result(),
                            ta*100))

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()

    return train_acc,epochs

def train_curve(train_acc,val_acc,epochs,name):

    plt.plot(epochs,train_acc,label="Training")
    plt.plot(epochs,val_acc,label="Validation")
    plt.xlabel("Training Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(name)

def main():
    # Invoke the above function to get our data.
    path = "/home/smathieson/Public/cs360/cifar-10-batches-py/"
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10(path)
    print('Train data shape: ', X_train.shape)              # (49000, 32, 32, 3)
    print('Train labels shape: ', y_train.shape)            # (49000,)
    print('Validation data shape: ', X_val.shape)           # (1000, 32, 32, 3)
    print('Validation labels shape: ', y_val.shape)         # (1000,)
    print('Test data shape: ', X_test.shape)                # (10000, 32, 32, 3)
    print('Test labels shape: ', y_test.shape)              # (10000,)

    ###### TODO: YOUR CODE HERE ######
    # set up train_dset, val_dset, and test_dset:
    # see documentation for tf.data.Dataset.from_tensor_slices, use batch = 64
    # train should be shuffled, but not validation and testing datasets
    train_dset = tf.data.Dataset.from_tensor_slices((X_train,y_train)).batch(batch_size=64)
    #train_dset = train_dset.batch(64)
    train_dset = train_dset.shuffle(len(X_train))
    val_dset = tf.data.Dataset.from_tensor_slices((X_val,y_val)).batch(64)

    test_dset = tf.data.Dataset.from_tensor_slices((X_test,y_test)).batch(64)

    for images,labels in train_dset:
        print(images.shape)
        print(labels.shape)

    print("pants")

    ######## END YOUR CODE #############

    ###### TODO: YOUR CODE HERE ######
    # call the train function to train a fully connected neural network
    #fc = FCmodel()
    #train_acc,val_acc,epochs = run_training(fc,train_dset,val_dset)
    #train_curve(train_acc,val_acc,epochs,"FCcurve.png")
    ######## END YOUR CODE #############

    ###### TODO: YOUR CODE HERE ######
    # call the train function to train a three-layer CNN
    cnn = CNNmodel()
    train_acc,val_acc,epochs = run_training(cnn,train_dset,val_dset)
    train_curve(train_acc,val_acc,epochs,"CNNcurve.png")

    confMat = np.zeros((10,10))
    for images,labels in test_dset:
        confMat = confusion_matrix(cnn,images,labels,confMat)

    print(confMat)
    ######## END YOUR CODE #############
