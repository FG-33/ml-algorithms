import unittest
import numpy as np
from shared.eval.evaluation import calc_precision_recall


class MyTestCase(unittest.TestCase):
    """ Check if recall and precision are correct, 8 predictions 0 tp 4 fp 4 fn 0 tn """
    def test_something1(self):

        y = np.array([1, 0, 1, 0, 1, 0, 1, 0]).reshape((-1, 1))
        prediction = np.array([0, 1, 0, 1, 0, 1, 0, 1]).reshape((-1, 1))

        precision, recall = calc_precision_recall(y, prediction)

        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)

    def test_something2(self):
        """ Check if recall and precision are correct, 3 predictions 3 tp 0 fp 0 fn 0 tn """
        y = np.array([1, 1, 1]).reshape((-1, 1))
        prediction = np.array([1, 1, 1]).reshape((-1, 1))

        precision, recall = calc_precision_recall(y, prediction)

        self.assertEqual(round(precision, 5), 1.0)
        self.assertEqual(round(recall, 5), 1.0)

    def test_something3(self):
        """ Check if recall and precision are correct, 5 predictions 2 tp 2 fp 1 fn 0 tn """
        y = np.array([1, 0, 1, 1, 0]).reshape((-1, 1))
        prediction = np.array([1, 1, 1, 0, 1]).reshape((-1, 1))

        precision, recall = calc_precision_recall(y, prediction)

        self.assertEqual(round(precision, 5), 0.5)
        self.assertEqual(round(recall, 5), 0.66667)


if __name__ == '__main__':
    unittest.main()
