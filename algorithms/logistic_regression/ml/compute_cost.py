import numpy as np

from logistic_regression.ml.sigmoid import sigmoid


def compute_cost(x, y, theta):
    """ Logistic regression cost function - Calculates the squared error
        between ground truth and predicted output

    :param x: input data samples x feature matrix
    :param y: output data or ground truth
    :param theta: logistic regression parameters learned
    :return predicted_result: squared error
    """
    m = y.shape[0]
    h = sigmoid(np.matmul(x, theta))
    return - 1/m * sum(y * np.log(h) + (1 - y) * np.log(1 - h))
