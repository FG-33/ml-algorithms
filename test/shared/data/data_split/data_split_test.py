import unittest
import numpy as np
from shared.data.data_split import train_eval_split


class MyTestCase(unittest.TestCase):
    def test_something1(self):
        """ Check if data is correctly being split if there are 35 samples with 5 input features and 1 output """
        num_samples = 35
        x = np.array([range(5) for i in range(num_samples)])
        y = np.array([range(1) for i in range(num_samples)])

        x_train, y_train, x_val, y_val = train_eval_split(x, y, 0.8)

        self.assertEqual(x_train.shape[0], y_train.shape[0])
        self.assertEqual(x_val.shape[0], y_val.shape[0])
        self.assertEqual(x_train.shape[0], 28)
        self.assertEqual(x_val.shape[0], 7)

    def test_something2(self):
        """ Check if data is correctly being split if there are 150 samples with 2 input features and 1 output """
        num_samples = 150
        x = np.array([range(2) for i in range(num_samples)])
        y = np.array([range(1) for i in range(num_samples)])

        x_train, y_train, x_val, y_val = train_eval_split(x, y, 0.2)

        self.assertEqual(x_train.shape[0], y_train.shape[0])
        self.assertEqual(x_val.shape[0], y_val.shape[0])
        self.assertEqual(x_train.shape[0], 30)
        self.assertEqual(x_val.shape[0], 120)


if __name__ == '__main__':
    unittest.main()
