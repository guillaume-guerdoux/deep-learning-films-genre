import numpy as np
import cv2
from scipy import misc


class DatasetManager:

    def __init__(self, training_set_dict, test_set_dict):
        self.training_set_dict = training_set_dict
        self.test_set_dict = test_set_dict
        self.training_samples_list = list(self.training_set_dict.keys())
        self.test_samples_list = list(self.test_set_dict.keys())
        self.cur_train = 0  # for train
        self.cur_test = 0  # for test
        self.absolute_path = 'assets/posters/'
        self.crop_size = 227
        self.mean = np.array([104., 117., 124.])

    def load_training_images(self, image_name_list, batch_size):
        images = np.ndarray([batch_size, self.crop_size, self.crop_size, 3])
        for index, image_name in enumerate(image_name_list):
            image = cv2.imread(
                self.absolute_path + self.training_set_dict[image_name] + '/' + image_name)
            image = cv2.resize(image, (self.crop_size, self.crop_size))
            image = image.astype(np.float32)
            image -= self.mean
            images[index] = image
        return images

    def load_test_images(self, image_name_list, batch_size):
        images = np.ndarray([batch_size, self.crop_size, self.crop_size, 3])
        for index, image_name in enumerate(image_name_list):
            image = cv2.imread(
                self.absolute_path + self.test_set_dict[image_name] + '/' + image_name)
            image = cv2.resize(image, (self.crop_size, self.crop_size))
            image = image.astype(np.float32)
            image -= self.mean
            images[index] = image
        return images

    def next_batch(self, batch_size, phase):
        # Get next batch of images and labels
        if phase == 'train':
            images = []
            if self.cur_train + batch_size < len(self.training_samples_list):
                images = self.load_training_images(
                    self.training_samples_list[self.cur_train:self.cur_train + batch_size], batch_size)
                # one_hot_labels = loaded_lab[self.cur_train:self.cur_train +
                # batch_size][:]
                self.cur_train += batch_size
            else:
                new_ptr = (self.cur_train + batch_size) % len(self.training_samples_list)
                images_list = self.training_samples_list[self.cur_train:] + self.training_samples_list[:new_ptr]
                images = self.load_training_images(images_list, batch_size)




                # one_hot_labels = np.concatenate((loaded_lab[self.cur_train:][:], loaded_lab[
                                            #    :(self.cur_train + batch_size) % self.train_size][:]), 0)
                self.cur_train = new_ptr
        elif phase == 'test':
            images = []
            if self.cur_test + batch_size < len(self.test_samples_list):
                images = self.load_test_images(
                    self.test_samples_list[self.cur_test:self.cur_test + batch_size], batch_size)
                # one_hot_labels = loaded_lab[self.cur_train:self.cur_train +
                # batch_size][:]
                self.cur_test += batch_size

            else:
                new_ptr = (self.cur_test + batch_size) % len(self.test_samples_list)
                images_list = self.test_samples_list[self.cur_test:] + self.test_samples_list[:new_ptr]
                images = self.load_test_images(images_list, batch_size)

                self.cur_test = new_ptr
        else:
            return None, None

        return images  #, one_hot_labels
