from numpy import matmul, power


def compute_cost(x, y, theta):
    """ Calculates the squared error between ground truth and predicted output

    :param x: input data samples x feature matrix
    :param y: output data or ground truth
    :param theta: linear regression parameters learned
    :return predicted_result: squared error
    """
    m = y.shape[0]
    return 1/(2 * m) * sum(power(matmul(x, theta) - y, 2))
