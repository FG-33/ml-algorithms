import unittest

from linear_regression.ml.predict import predict
from numpy import array


class MyTestCase(unittest.TestCase):
    def test_something1(self):
        """ Predict one sample with two features (and theta0)"""
        x, y, theta = array([1, 2, 3]), array([14]), array([1, 2, 3])
        y_predicted = predict(x, theta)
        self.assertEqual(y, y_predicted)

    def test_something2(self):
        """ Predict two samples with two features (and theta0)"""
        x, y, theta = array([[1, 2, 3], [1, 2, 4]]), array([[14], [17]]), array([1, 2, 3])
        y_predicted = predict(x, theta)
        self.assertEqual(y[0], y_predicted[0])
        self.assertEqual(y[1], y_predicted[1])


if __name__ == '__main__':
    unittest.main()
