from numpy import array, expand_dims
from pandas import read_csv


def load_data(path):
    """ Load file and return extract data. Return data as two numpy arrays

    :param path: relative/absolute path to the file to load
    :return x: Input matrix -> m x n (m = samples, n = features)
    :return y: Output vector -> m x 1 (m = see above)
    """
    data = read_csv(path, header=None)
    input_cols = data.columns[0:-1]
    output_col = data.columns[-1]
    x, y = array(data[input_cols]), array(data[output_col])

    if len(x.shape) == 1:
        x = expand_dims(x, axis=1)
    y = expand_dims(y, axis=1)

    return x, y
