import unittest
import numpy as np
from utils import *


class MeanAveragePrecisionTests(unittest.TestCase):

    def test_mean_average_precision_equal_to_one(self):
        outputs = np.array(([0.12, 0.11, 0.45], [0.1, 0.9, 0.87],
                           [0.45, 0.65, 0.8]))
        labels = np.array(([0, 0, 1], [0, 1, 1],
                          [1, 1, 1]))
        self.assertEqual(mean_average_precision(outputs, labels), 1)
