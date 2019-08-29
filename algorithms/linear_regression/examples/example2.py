from linear_regression.const import HEIGHT_WEIGHT_CATHETER_PATH
from linear_regression.ml.gradient_descent import gradient_descent
from linear_regression.util.load_data import load_data
from linear_regression.util.plot import plot

# load data
x, y = load_data(HEIGHT_WEIGHT_CATHETER_PATH())

# train
theta, costs = gradient_descent(x, y, alpha=0.0005, num_iters=150)

print("theta: {}".format(theta))
print("final cost: {}".format(costs[-1]))

# plot cost function over time
data = [(range(len(costs)), costs)]
labels = ("Iterations", "Cost J")
plot(2, False, labels, data, ["b-"])
