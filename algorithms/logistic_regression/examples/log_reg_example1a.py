import numpy as np
from logistic_regression.const import MUSHROOMS_PATH
from shared.eval.evaluation import calc_precision_recall
from logistic_regression.ml.gradient_descent_with_eval import gradient_descent
from logistic_regression.ml.predict import predict_with_thresh
from shared.data.load_data import load_data
from shared.data.plot import plot
from shared.data.data_split import train_eval_split

# set random seed
np.random.seed(10)

# load data
x, y = load_data(MUSHROOMS_PATH())

# train, eval split
x_train, y_train, x_val, y_val = train_eval_split(x, y, 0.7)
# print(x_train.shape)  # 6498

# train
theta, costs, train_evals, eval_evals = gradient_descent(x_train, y_train, x_val, y_val, alpha=0.5, num_iters=300)

# plot cost function over time
data = [(range(len(costs)), costs)]
labels = ("Iterations", "Cost J")
plot(2, True, labels, data, ["b-"])

data = [(range(len(train_evals)), [t[0] for t in train_evals]), (range(len(eval_evals)), [e[0]for e in eval_evals])]
plot(3, True, ("Steps", "Precision"), data, ["b-", "r-"])

data = [(range(len(train_evals)), [e[1] for e in train_evals]), (range(len(eval_evals)), [e[1] for e in eval_evals])]
plot(4, False, ("Steps", "Recall"), data, ["b-", "r-"])


# eval
prediction = predict_with_thresh(x_val, theta, 0.7)
recall, precision = calc_precision_recall(y_val, prediction)
