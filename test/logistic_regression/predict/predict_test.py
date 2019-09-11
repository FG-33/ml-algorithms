import unittest
import numpy as np
from logistic_regression.ml.predict import predict, predict_with_thresh


class MyTestCase(unittest.TestCase):
    def test_something1(self):
        """ Predict one sample with two features (and theta0)"""
        x, y, theta = np.array([1, 2, 3]), np.array([1.]), np.array([1, 2, 3])
        y_predicted = predict(x, theta)
        self.assertEqual(y, np.round(y_predicted, 5))

    def test_something2(self):
        """ Predict two samples with two features (and theta0)"""
        x, y, theta = np.array([[1, 2, 3], [-1, -2, -4]]), np.array([[1.], [.0]]), np.array([1, 2, 3])
        y_predicted = predict(x, theta)
        self.assertEqual(y[0], np.round(y_predicted[0], 5))
        self.assertEqual(y[1], np.round(y_predicted[1], 5))

    def test_something3(self):
        """ Predict one sample with two features (and theta0)"""
        x, y, theta = np.array([1, 2, 3]), np.array([[1.]]), np.array([1, 2, 3])
        y_predicted = predict_with_thresh(x, theta, 0.5)
        self.assertEqual(y, y_predicted)

    def test_something4(self):
        """ Predict two samples with two features (and theta0)"""
        x, y, theta = np.array([[1, 2, 3], [-1, -2, -4]]), np.array([[1.], [.0]]), np.array([1, 2, 3])
        y_predicted = predict_with_thresh(x, theta, 0.5)
        self.assertEqual(y[0], y_predicted[0])
        self.assertEqual(y[1], y_predicted[1])


if __name__ == '__main__':
    unittest.main()
