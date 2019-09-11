import unittest
import numpy as np
from logistic_regression.ml.compute_cost import compute_cost


class MyTestCase(unittest.TestCase):
    def test_something1(self):
        """ compute cost with a single feature and only one sample
        (theta0 included -> y = theta0 + x1 * theta1)"""
        x, y, theta = np.array([1, -1]), np.array([0]), np.array([[-10], [10]])
        self.assertEqual(0, np.round(compute_cost(x, y, theta), 5))

    def test_something2(self):
        """ compute cost with a single feature and only one sample
        (theta0 not included included -> y = x1 * theta1)"""
        x, y, theta = np.array([5]), np.array([1]), np.array([4])
        self.assertEqual(0, np.round(compute_cost(x, y, theta), 5))

    def test_something3(self):
        """ compute cost with a single feature and only one sample
        (theta0 included -> y = theta0 + x1 * theta1)"""
        x, y, theta = np.array([1, -1]), np.array([1]), np.array([[-10], [10]])
        self.assertGreater(np.round(compute_cost(x, y, theta), 5), 15)


if __name__ == '__main__':
    unittest.main()
