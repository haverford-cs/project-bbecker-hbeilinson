"""
Fully connected neural network architecture.
Author:
Date:
"""

import numpy as np
import tensorflow as tf
import sys

# from tensorflow.keras.layers import Dense, Flatten, Conv2D
# from tensorflow.keras import Model

##################

class FCmodel(Model):
    """
    A fully-connected neural network; the architecture is:
    fully-connected (dense) layer -> ReLU -> fully connected layer.
    Note that we only need to define the forward pass here; TensorFlow will take
    care of computing the gradients for us.
    """
    def __init__(self):
        super(FCmodel, self).__init__()

        # TODO set up architecture, for example:
        self.d0 = Flatten()
        self.d1 = Dense(4000, activation = "relu")
        self.d2 = Dense(10,activation="softmax")
        # use 4000 units in the hidden layer, num classes is 10

    def call(self, x):

        # TODO apply each layer from the constructor to x, returning
        # the output of the last layer
        x = self.d0(x)
        print(x)
        x = self.d1(x)
        return self.d2(x)


def two_layer_fc_test():
    """Test function to make sure the dimensions are working"""

    # Create an instance of the model
    fc_model = FCmodel()

    # TODO try out both the options below (all zeros and random)
    # shape is: number of examples (mini-batch size), width, height, depth
    x_np = np.zeros((64, 32, 32, 3))
    #x_np = np.random.rand(64, 32, 32, 3)

    # call the model on this input and print the result
    output = fc_model.call(x_np)
    print(output) # TODO what shape is this? does it make sense?
    # TODO look at the model parameter shapes, do they make sense?
    for v in fc_model.trainable_variables:
        print("Variable:", v.name)
        print("Shape:", v.shape)

def main():
    # test two layer function
    two_layer_fc_test()

if __name__ == "__main__":
    main()
