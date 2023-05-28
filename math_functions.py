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
    return dout * (x >= 0)


def relu_derivative2(x, dout):
    return np.where(x > 0, dout, 0)


def softmax(x):
    ex = np.exp(x)
    return ex / np.sum(ex, axis=0)


def mean_squared_error(y_true, y_pred):
    return np.square(np.subtract(y_true, y_pred)).mean(axis=0)


def mean_squared_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_derivative(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
