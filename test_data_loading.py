import unittest
import pickle
from dataset_manager import DatasetManager


class DataLoadingTests(unittest.TestCase):
    def setUp(self):
        with open('training_set_dict.pickle', 'rb') as handle:
            self.training_dict = pickle.load(handle)
        with open('test_set_dict.pickle', 'rb') as handle:
            self.test_dict = pickle.load(handle)
        self.dataset_manager = DatasetManager(self.training_dict,
                                              self.test_dict)

        self.batch_size = 50

    def test_normal_training_image_load(self):
        images = self.dataset_manager.next_batch(50, "train")
        self.assertEqual(images.shape, (50, 227, 227, 3))

    def test_last_traninig_image_load(self):
        self.dataset_manager.cur_train = \
            len(self.dataset_manager.training_samples_list) - \
            (self.batch_size / 2)
        images = self.dataset_manager.next_batch(50, "test")
        self.assertEqual(images.shape, (50, 227, 227, 3))

    def test_normal_test_image_load(self):
        images = self.dataset_manager.next_batch(50, "test")
        self.assertEqual(images.shape, (50, 227, 227, 3))

    def test_last_test_image_load(self):
        self.dataset_manager.cur_tests = \
            len(self.dataset_manager.test_samples_list) - \
            (self.batch_size / 2)
        images = self.dataset_manager.next_batch(50, "test")
        self.assertEqual(images.shape, (50, 227, 227, 3))
