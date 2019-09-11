def train_eval_split(x, y, perc):
    """ Split given data into training and evaluation sets.

    :param x: input data
    :param y: output data
    :param perc: Specifies amount of training data in percentage from total (0.7 -> 70% train, 30% eval)
    :return x_train, y_train, x_val, y_val: new sets used for training and evaluation
    """
    idx_split = round(x.shape[0] * perc)
    x_train, y_train = x[:idx_split, :], y[:idx_split, :]
    x_val, y_val = x[idx_split:, :], y[idx_split:, :]

    return x_train, y_train, x_val, y_val
