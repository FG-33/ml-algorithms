import unittest
from numpy import array
from ml.compute_cost import compute_cost


class MyTestCase(unittest.TestCase):
    def test_something1(self):
        """ compute cost with a single feature and only one sample
        (theta0 included -> y = theta0 + x1 * theta1)"""
        x, y, theta = array([1, 1]), array([0]), array([[0], [0]])
        self.assertEqual(0, compute_cost(x, y, theta))

    def test_something2(self):
        """ compute cost with a single feature and only one sample
        (theta0 not included included -> y = x1 * theta1)"""
        x, y, theta = array([5]), array([15]), array([4])
        self.assertEqual(12.5, compute_cost(x, y, theta))

    def test_something3(self):
        """ compute cost with a single feature and two samples
        (theta0 included)"""
        x, y, theta = array([[1, 5], [1, 7]]), array([[17], [32]]), array([[3], [3]])
        self.assertEqual(16.25, compute_cost(x, y, theta))

    def test_something4(self):
        """ compute cost with a five features and a single sample
        (theta0 included)"""
        x, y, theta = array([1, 1, 3, 4, 5, 6]), array([20]), array([[-1], [0.5], [2], [-1], [-2], [3]])
        self.assertEqual(55.125, compute_cost(x, y, theta))

    def test_something5(self):
        """ compute cost with a three features and two samples
        (theta0 included)"""
        x, y, theta = array([[1, 1, 3, 6], [1, 5, 2, 3]]), array([[20], [10]]), array([[-1], [0.5], [2], [3]])
        self.assertEqual(8.125, compute_cost(x, y, theta))


if __name__ == '__main__':
    unittest.main()
