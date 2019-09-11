import unittest
import numpy as np
from logistic_regression.ml.gradient_descent import gradient_descent


class MyTestCase(unittest.TestCase):
    def test_something1(self):
        """ Check whether or not num_iters iterations were run """
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape((-1, 1))
        y = np.array([2, 4, 6, 8, 10, 12, 14, 16]).reshape((-1, 1))

        num_iters = 100
        theta, costs = gradient_descent(x, y, alpha=0.01, num_iters=num_iters)
        self.assertEqual(len(costs), num_iters)

    def test_something2(self):
        """ Check if cost does always decrease/equal """
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape((-1, 1))
        y = np.array([2, 4, 6, 8, 10, 12, 14, 16]).reshape((-1, 1))

        num_iters = 100
        theta, costs = gradient_descent(x, y, alpha=0.01, num_iters=100)

        for i in range(num_iters - 1):
            self.assertGreaterEqual(np.round(costs[i], 5), np.round(costs[i + 1], 5))

    def test_something3(self):
        """ Check if cost does always decrease/equal """
        x = np.array([1, 4, 7, 24, 45, 66, 87, 118]).reshape((-1, 1))
        y = np.array([2, 4, 9, 8, 17, 12, 14, 20]).reshape((-1, 1))

        num_iters = 100
        theta, costs = gradient_descent(x, y, alpha=0.0001, num_iters=num_iters)

        for i in range(num_iters - 1):
            self.assertGreaterEqual(np.round(costs[i], 5), np.round(costs[i + 1], 5))

    def test_something4(self):
        """ Check if theta has exactly len(x) entries (it is to be assumed that x
            contains a leading one it theta_0 is desired) """
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape((-1, 1))
        y = np.array([2, 4, 6, 8, 10, 12, 14, 16]).reshape((-1, 1))

        theta, costs = gradient_descent(x, y, alpha=0.01, num_iters=100)

        self.assertEqual(theta.shape[0], x.shape[1])

    def test_something5(self):
        """ Check if theta has exactly x.shape[1] entries (it is to be assumed that x
            contains a leading one it theta_0 is desired) """
        x = np.array([[1, 1, 5], [2, 4, 1], [3, 6, 2], [4, 1, 9], [5, 4, 3], [6, 8, 6], [7, 9, 1], [8, 9, 8]])
        y = np.array([2, 4, 6, 8, 10, 12, 14, 16]).reshape((-1, 1))

        theta, costs = gradient_descent(x, y, alpha=0.01, num_iters=100)

        self.assertEqual(theta.shape[0], x.shape[1])

    def test_something6(self):
        """ Check if theta is similar to the function that'd produce these results -> y = sigmoid(x * 2.5) """
        x = np.array([1, 2, 3, 4, -1, -2, -3, -4]).reshape((-1, 1))
        y = np.array([1, 1, 1, 1, 0, 0, 0, 0]).reshape((-1, 1))

        theta, costs = gradient_descent(x, y, alpha=0.01, num_iters=5000)

        # should be around 2.5 so that sigmoid(..) returns ~ 1 for positive numbers
        self.assertGreater(np.round(theta[0], 5), np.round(2.5, 5))

    def test_something7(self):
        """ Check if theta is similar to the function that'd produce these results -> y = sigmoid(x * -2.5) """
        x = np.array([-1, -2, -3, -4, 1, 2, 3, 4]).reshape((-1, 1))
        y = np.array([1, 1, 1, 1, 0, 0, 0, 0]).reshape((-1, 1))

        theta, costs = gradient_descent(x, y, alpha=0.01, num_iters=5000)

        # should be around -2.5 so that sigmoid(..) returns ~ 0 for positive numbers
        self.assertLess(np.round(theta[0], 5), np.round(-2.5, 5))


if __name__ == '__main__':
    unittest.main()
