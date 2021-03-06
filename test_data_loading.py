import unittest
import pickle
import json
from dataset_manager import DatasetManager


class DataLoadingTests(unittest.TestCase):
    def setUp(self):
        with open('training_set_list.pickle', 'rb') as handle:
            self.training_dict = pickle.load(handle)
        with open('validation_set_list.pickle', 'rb') as handle:
            self.validation_dict = pickle.load(handle)
        with open('test_set_list.pickle', 'rb') as handle:
            self.test_dict = pickle.load(handle)
        with open('genres.json') as json_data:
            self.genres = json.load(json_data)
        with open('labels.json') as json_data:
            self.dataset = json.load(json_data)
        self.dataset_manager = DatasetManager(self.training_dict,
                                              self.validation_dict,
                                              self.test_dict,
                                              self.genres,
                                              self.dataset)
        self.batch_size = 50

    def test_normal_training_image_load(self):
        images = self.dataset_manager.next_batch(50, "train")
        self.assertEqual(images[0].shape, (50, 227, 227, 3))

    def test_normal_training_labels_load(self):
        images = self.dataset_manager.next_batch(50, "train")
        self.assertEqual(images[1].shape, (50, 26))

    def test_last_traninig_image_load(self):
        self.dataset_manager.cur_train = \
            len(self.dataset_manager.training_list) - \
            (self.batch_size / 2)
        images = self.dataset_manager.next_batch(50, "train")
        self.assertEqual(images[0].shape, (50, 227, 227, 3))

    def test_last_traninig_labels_load(self):
        self.dataset_manager.cur_train = \
            len(self.dataset_manager.training_list) - \
            (self.batch_size / 2)
        images = self.dataset_manager.next_batch(50, "train")
        self.assertEqual(images[1].shape, (50, 26))

    def test_normal_validation_image_load(self):
        images = self.dataset_manager.next_batch(50, "val")
        self.assertEqual(images[0].shape, (50, 227, 227, 3))

    def test_normal_validation_labels_load(self):
        images = self.dataset_manager.next_batch(50, "val")
        self.assertEqual(images[1].shape, (50, 26))

    def test_last_validation_image_load(self):
        self.dataset_manager.cur_val = \
            len(self.dataset_manager.validation_list) - \
            (self.batch_size / 2)
        images = self.dataset_manager.next_batch(50, "val")
        self.assertEqual(images[0].shape, (50, 227, 227, 3))

    def test_last_validation_labels_load(self):
        self.dataset_manager.cur_val = \
            len(self.dataset_manager.validation_list) - \
            (self.batch_size / 2)
        images = self.dataset_manager.next_batch(50, "val")
        self.assertEqual(images[1].shape, (50, 26))

    def test_normal_test_image_load(self):
        images = self.dataset_manager.next_batch(50, "test")
        self.assertEqual(images[0].shape, (50, 227, 227, 3))

    def test_normal_test_labels_load(self):
        images = self.dataset_manager.next_batch(50, "test")
        self.assertEqual(images[1].shape, (50, 26))

    def test_last_test_image_load(self):
        self.dataset_manager.cur_test = \
            len(self.dataset_manager.test_list) - \
            (self.batch_size / 2)
        images = self.dataset_manager.next_batch(50, "test")
        self.assertEqual(images[0].shape, (50, 227, 227, 3))

    def test_last_test_labels_load(self):
        self.dataset_manager.cur_test = \
            len(self.dataset_manager.test_list) - \
            (self.batch_size / 2)
        images = self.dataset_manager.next_batch(50, "test")
        self.assertEqual(images[1].shape, (50, 26))

    def test_create_label_vector(self):
        label_vector = self.dataset_manager.create_label_vector(
            [" Action", " Documentary",
             " Drama", " Horror", " News", " War"])
        self.assertEqual(label_vector, [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                                        1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0])

    def test_create_label_vector_end(self):
        label_vector = self.dataset_manager.create_label_vector(
            [" Action", " Documentary",
             " Drama", " Horror", " News", " War", " Western"])
        self.assertEqual(label_vector, [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                                        1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1])

    def test_no_duplicate_between_test_and_train(self):
        self.assertEqual(self.training_dict.intersection(self.test_dict), set())
