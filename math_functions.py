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
    x_max = np.max(x)
    sub = np.subtract(x, x_max)
    ex = np.exp(sub)
    return ex / np.sum(ex)


def softmax2(x):
    ex = np.exp(x)
    sum = np.sum(ex, axis=0)
    if sum == 0:
        return np.zeros()
    return ex / sum


def mean_squared_error(y_true, y_pred):
    return np.square(np.subtract(y_true, y_pred)).mean(axis=0)


def mean_squared_error_derivative(y_true, y_pred):
    # return (2 * (y_pred - y_true)) / np.size(y_true)
    return 2 / np.size(y_true) * (y_pred - y_true)


def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_derivative(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


def nll_loss(y_true, y_pred):
    np.clip(y_pred, 1e-10, 1. - 1e-10)
    loss = -np.sum(y_true * np.log(y_pred))
    return loss