import numpy as np
from util.load_data import load_data
from ml.gradient_descent import gradient_descent
from util.plot import plot
from ml.predict import predict
from const import BODY_BRAIN_PATH

# load data
x, y = load_data(BODY_BRAIN_PATH())

# add 1s to x
m, n = x.shape

# train
theta, costs = gradient_descent(x, y, alpha=0.0000002, num_iters=150)

# plot data with result line
_x = np.array([0.0, 10.0, 100.0, 1000.0, 8000.0]).reshape(5, 1)
_y = predict(_x, theta)
data = [(x, y), (_x, _y)]
labels = ("Brain weight", "Body weight")
plot(1, True, labels, data, ["rx", "b-"])

# plot cost function over time
data = [(range(len(costs)), costs)]
labels = ("Iterations", "Cost J")
plot(2, False, labels, data, ["b-"])
