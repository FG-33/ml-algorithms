import unittest
from logistic_regression.ml.sigmoid import sigmoid
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something1(self):
        """ Use sigmoid function on a single value """
        result = sigmoid(1.5)
        self.assertEqual(0.817574476, np.round(result, 9))

    def test_something2(self):
        """ Use sigmoid function on a 1x1 numpy np.array """
        result = sigmoid(np.array([1.5]))
        self.assertEqual(0.817574476, np.round(result, 9))

    def test_something3(self):
        """ Use sigmoid function on a 2x1 numpy np.array """
        result = sigmoid(np.array([[1.5], [2.5]]))
        control, actual = np.array([[0.817574476], [0.92414182]]), np.round(result, 9)
        self.assertEqual(True, (control == actual).all())

    def test_something4(self):
        """ Use sigmoid function on a 2x3 numpy np.array """
        result = sigmoid(np.array([[1.5, 3, 2], [2.5, 5, -2]]))
        control = np.array([[0.817574476, 0.952574127, 0.880797078], [0.92414182, 0.993307149, 0.119202922]])
        actual = np.round(result, 9)
        self.assertEqual(True, (control == actual).all())

    def test_something5(self):
        """ sigmoid -> 1 if x -> inf """
        result = sigmoid(1000000000)
        self.assertEqual(np.round(result, 5), 1.0)

    def test_something6(self):
        """ sigmoid -> 0 if x -> -inf """
        result = sigmoid(-1000000000)
        self.assertEqual(np.round(result, 5), 0.0)


if __name__ == '__main__':
    unittest.main()
