import unittest

from linear_regression.util.load_data import load_data


class MyTestCase(unittest.TestCase):
    def test_something1(self):
        """ Check if load_data correctly extracts the 5 row 2 column data from testdata1.txt """
        x, y = load_data("testdata1.txt")
        self.assertEqual((5, 1), x.shape)
        self.assertEqual((5, 1), y.shape)

    def test_something2(self):
        """ Check if load_data correctly extracts the 5 row 3 column data from testdata1.txt """
        x, y = load_data("testdata2.txt")
        self.assertEqual((5, 2), x.shape)
        self.assertEqual((5, 1), y.shape)


if __name__ == '__main__':
    unittest.main()
