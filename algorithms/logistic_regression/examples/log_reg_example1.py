import numpy as np
from logistic_regression.const import MUSHROOMS_PATH
from shared.eval.evaluation import calc_precision_recall
from logistic_regression.ml.gradient_descent import gradient_descent
from logistic_regression.ml.predict import predict_with_thresh
from shared.data.load_data import load_data
from shared.data.plot import plot
from shared.data.data_split import train_eval_split

# set random seed
np.random.seed(1)

# load data
x, y = load_data(MUSHROOMS_PATH())

# train, eval split
x_train, y_train, x_val, y_val = train_eval_split(x, y, 0.2)

# train
theta, costs = gradient_descent(x_train, y_train, alpha=0.5, num_iters=5000)

# plot cost function over time
data = [(range(len(costs)), costs)]
labels = ("Iterations", "Cost J")
plot(2, False, labels, data, ["b-"])


# eval
prediction = predict_with_thresh(x_val, theta, 0.7)
recall, precision = calc_precision_recall(y_val, prediction)
