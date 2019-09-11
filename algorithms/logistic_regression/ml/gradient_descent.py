import numpy as np
from logistic_regression.ml.compute_cost import compute_cost
from logistic_regression.ml.sigmoid import sigmoid


def gradient_descent(x, y, theta=None, alpha=0.01, num_iters=100):
    """ Performs gradient descent to determine the best possible model based on given input data (x)
        output data (y)

    :param x: input data samples x feature matrix
    :param y: output data or ground truth
    :param theta: initial logistic regression parameters
    :param alpha: learning rate
    :param num_iters: number of iterations to run gradient descent
    :return theta: learned logistic regression parameters (parameters of the  model)
    :return costs: list containing the calculated cost values for each step of gradient descent
    """
    m, n = x.shape
    if theta is None:
        theta = np.random.rand(n, 1)

    costs = []
    for i in range(num_iters):
        theta -= alpha * (1 / m) * np.matmul(np.transpose(x), (sigmoid(np.matmul(x, theta)) - y))
        costs.append(compute_cost(x, y, theta))

    return theta, costs
