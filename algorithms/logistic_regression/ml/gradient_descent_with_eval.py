import numpy as np
from logistic_regression.ml.compute_cost import compute_cost
from logistic_regression.ml.predict import predict_with_thresh
from logistic_regression.ml.sigmoid import sigmoid
from shared.eval.evaluation import calc_precision_recall


def gradient_descent(x_train, y_train, x_val, y_val, theta=None, alpha=0.01, num_iters=100, val_thresh=0.7):
    """ Performs gradient descent to determine the best possible model based on given input data (x)
        output data (y).

    :param x_train: input data samples x feature matrix
    :param y_train: output data or ground truth
    :param x_val: input data samples x feature matrix (evaluation)
    :param y_val: output data or ground truth (evaluation)
    :param theta: initial logistic regression parameters
    :param alpha: learning rate
    :param num_iters: number of iterations to run gradient descent
    :param val_thresh: threshold for predictions 1/0
    :return theta: learned logistic regression parameters (parameters of the  model)
    :return costs: list containing the calculated cost values for each step of gradient descent
    """
    m, n = x_train.shape
    if theta is None:
        theta = np.random.rand(n, 1)

    costs = []
    train_evals = []
    eval_evals = []

    # # # with random theta
    # eval val
    prediction = predict_with_thresh(x_val, theta, val_thresh)
    recall, precision = calc_precision_recall(y_val, prediction)
    eval_evals.append((precision, recall))

    # train val
    prediction = predict_with_thresh(x_train, theta, val_thresh)
    recall, precision = calc_precision_recall(y_train, prediction)
    train_evals.append((precision, recall))

    for i in range(num_iters):
        theta -= alpha * (1 / m) * np.matmul(np.transpose(x_train), (sigmoid(np.matmul(x_train, theta)) - y_train))
        costs.append(compute_cost(x_train, y_train, theta))

        # eval val
        prediction = predict_with_thresh(x_val, theta, val_thresh)
        recall, precision = calc_precision_recall(y_val, prediction)
        eval_evals.append((precision, recall))

        # train val
        prediction = predict_with_thresh(x_train, theta, val_thresh)
        recall, precision = calc_precision_recall(y_train, prediction)
        train_evals.append((precision, recall))

    return theta, costs, train_evals, eval_evals
