import numpy as np


def calc_precision_recall(y, prediction):
    """ Evaluates the prediction given a specific theta. Calculates Recall and precision based on ground
    truth output data and predicted output data

    :param y: output data or ground truth
    :param prediction: predicted output
    :return precision: Specifies how many of the positive outputs is indeed positive
    :return recall: Specifies how many of the positive outputs were correctly classified by theta
    """
    tp_, fp_, fn_ = np.zeros(y.shape), np.zeros(y.shape), np.zeros(y.shape)

    tp_[(y == 1) * (prediction == 1)] = 1
    tp = sum(tp_)

    fp_[(y == 0) * (prediction == 1)] = 1
    fp = sum(fp_)

    fn_[(y == 1) * (prediction == 0)] = 1
    fn = sum(fn_)

    precision = tp / (tp + fp + 0.000000000001)
    recall = tp / (tp + fn + 0.000000000001)

    print("{}% Recall".format(round(recall[0] * 100, 1)))
    print("{}% Precision".format(round(precision[0] * 100, 1)))

    return precision[0], recall[0]
