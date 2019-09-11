import numpy as np
from logistic_regression.ml.sigmoid import sigmoid


def predict(x, theta):
    """ Predicts outputs based on input samples/features given theta

    :param x: input data samples x feature matrix
    :param theta: linear regression parameters learned
    :return predicted_result: predicted output for given input data
    """
    return sigmoid(np.matmul(x, theta))


def predict_with_thresh(x, theta, thresh):
    """ Predicts outputs based on input samples/features given theta. Also applies a threshold
    Values over threshold -> 1
    Values under threshold -> 0

    :param x: input data samples x feature matrix
    :param theta: linear regression parameters learned
    :param thresh: specifies whether a value is an indicator for positive (1) or negative (0)
    :return predicted_result: predicted output for given input data
    """
    prediction = sigmoid(np.matmul(x, theta))
    if np.isscalar(prediction):
        return 1 if prediction >= thresh else 0

    prediction[prediction < thresh] = 0
    prediction[prediction >= thresh] = 1
    return prediction
