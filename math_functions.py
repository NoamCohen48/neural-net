import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x, dout):
    sig = sigmoid(x)
    return sig * (1 - sig) * dout


def relu(x):
    return x * (x > 0)


def relu2(x):
    """
    IN PLACE
    :param x:
    :return:
    """
    return np.maximum(x, 0, x)


def relu_derivative(x, dout):
    # np.maximum(x, 0)
    return dout * (x >= 0)


def relu_derivative2(x, dout):
    """

    :param x:
    :param dout: derivative to pass if weight is positive
    :return:
    """
    dx = np.where(x > 0, dout, 0)
    return dx


def softmax(x):
    ex = np.exp(x)
    return ex / np.sum(ex)
