import numpy as np
import cv2
from image_processing import *
import random


class DatasetManager:

    def __init__(self, training_set, test_set, genres, labels):
        self.training_list = list(training_set)
        self.test_list = list(test_set)
        self.genres = genres
        self.labels = labels
        # self.training_samples_list = list(self.training_set_dict.keys())
        # self.test_samples_list = list(self.test_set_dict.keys())
        self.cur_train = 0  # for train
        self.cur_test = 0  # for test
        self.absolute_path = 'assets/posters/'
        self.resize_size = 227
        # RGB values of the mean-image of ILSVRC
        self.mean = np.array([104., 117., 124.])

    def load_training_images(self, image_name_list, batch_size):
        images = np.ndarray([batch_size, self.resize_size, self.resize_size, 3])
        for index, image_name in enumerate(image_name_list):
            img = cv2.imread(
                self.absolute_path + image_name)
            # 1/3 change to get each transformation
            crop_chance = random.random()
            blur_chance = random.random()
            rotate_chance = random.random()
            rotate_zoom_chance = random.random()
            translation_chance = random.random()
            if crop_chance <= 0.33:
                img = random_crop(img)
            if blur_chance <= 0.33:
                img = blur(img, random.choice([5, 7, 15]))
            if rotate_zoom_chance <= 0.33:
                img = random_rotate(img)
            elif rotate_chance <= 0.33:
                img = random_rotate_zoom(img)
            if translation_chance <= 0.33:
                img = random_translate(img)
            img = cv2.resize(img, (self.resize_size, self.resize_size))
            img = img.astype(np.float32)
            img -= self.mean
            images[index] = img
        return images

    # TODO : Try 0.99 and 0.01
    def create_label_vector(self, label_list):
        label_vector = [0]*len(self.genres)
        for label in label_list:
            label_vector[self.genres.index(label)] = 1
        return label_vector

    def load_training_labels(self, image_name_list, batch_size):
        labels_outputs = np.ndarray([batch_size, len(self.genres)])
        for index, image_name in enumerate(image_name_list):
            image_genre = self.labels[image_name]
            label_vector = self.create_label_vector(image_genre)
            labels_outputs[index] = label_vector
        return labels_outputs

    def load_test_images(self, image_name_list, batch_size):
        images = np.ndarray([batch_size, self.resize_size, self.resize_size, 3])
        for index, image_name in enumerate(image_name_list):
            img = cv2.imread(
                self.absolute_path + '/' + image_name)
            # 1/3 change to get each transformation
            crop_chance = random.random()
            blur_chance = random.random()
            rotate_chance = random.random()
            rotate_zoom_chance = random.random()
            translation_chance = random.random()
            if crop_chance <= 0.33:
                img = random_crop(img)
            if blur_chance <= 0.33:
                img = blur(img, random.choice([5, 7, 15]))
            if rotate_zoom_chance <= 0.33:
                img = random_rotate(img)
            elif rotate_chance <= 0.33:
                img = random_rotate_zoom(img)
            if translation_chance <= 0.33:
                img = random_translate(img)
            img = cv2.resize(img, (self.resize_size, self.resize_size))
            img = img.astype(np.float32)
            img -= self.mean
            images[index] = img
        return images

    def load_test_labels(self, image_name_list, batch_size):
        labels_outputs = np.ndarray([batch_size, len(self.genres)])
        for index, image_name in enumerate(image_name_list):
            image_genre = self.labels[image_name]
            label_vector = self.create_label_vector(image_genre)
            labels_outputs[index] = label_vector
        return labels_outputs

    def next_batch(self, batch_size, phase):
        # Get next batch of images and labels
        if phase == 'train':
            images = []
            if self.cur_train + batch_size < len(self.training_list):
                # print("train normal")
                images = self.load_training_images(
                    self.training_list[int(self.cur_train):int(self.cur_train + batch_size)], batch_size
                )
                labels = self.load_training_labels(
                    self.training_list[int(self.cur_train):int(self.cur_train + batch_size)], batch_size
                )
                # batch_size][:]
                self.cur_train += batch_size
            else:
                # print("train last")
                new_ptr = (self.cur_train + batch_size) % len(self.training_list)
                images_list = self.training_list[int(self.cur_train):] + \
                    self.training_list[:int(new_ptr)]
                images = self.load_training_images(images_list, batch_size)
                labels = self.load_training_labels(images_list, batch_size)
                self.cur_train = new_ptr
        elif phase == 'test':
            images = []
            if self.cur_test + batch_size < len(self.test_list):
                # print("test normal")
                images = self.load_test_images(
                    self.test_list[int(self.cur_test):int(self.cur_test + batch_size)], batch_size)
                labels = self.load_test_labels(
                    self.test_list[int(self.cur_train):int(self.cur_train + batch_size)], batch_size
                )
                self.cur_test += batch_size

            else:
                # print("test last")
                new_ptr = (self.cur_test + batch_size) % len(self.test_list)
                images_list = self.test_list[int(self.cur_test):] + \
                    self.test_list[:int(new_ptr)]
                images = self.load_test_images(images_list, batch_size)
                labels = self.load_test_labels(images_list, batch_size)
                self.cur_test = new_ptr
        else:
            return None, None

        return images, labels
