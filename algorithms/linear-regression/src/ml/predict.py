from numpy import matmul


def predict(x, theta):
    """ Predicts outputs based on input samples/features given theta

    :param x: input data samples x feature matrix
    :param theta: linear regression parameters learned
    :return predicted_result: predicted output for given input data
    """
    return matmul(x, theta)
