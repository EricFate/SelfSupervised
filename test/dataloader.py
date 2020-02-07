import unittest
from data import *
import matplotlib.pyplot as plt


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = GenericDataset('tinyimagenet', 'train')

    def test_jigsaw(self):
        loader = JigsawDataLoader(self.dataset,2)
        img, label = next(iter(loader()))
        self.assertTrue(True, False)


if __name__ == '__main__':
    unittest.main()
