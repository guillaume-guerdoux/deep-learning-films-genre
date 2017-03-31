import tensorflow as tf
import pickle
import json
import cv2
import numpy as np

from dataset_manager import DatasetManager
from model import Model
from network import load_with_skip
from utils import mean_average_precision
import time


def main():
    # Load dataset manager
    with open('training_set_list.pickle', 'rb') as handle:
        training_set = pickle.load(handle)
    with open('validation_set_list.pickle', 'rb') as handle:
        validation_set = pickle.load(handle)
    with open('test_set_list.pickle', 'rb') as handle:
        test_set = pickle.load(handle)
    with open('genres.json') as json_data:
        genres = json.load(json_data)
    with open('labels.json') as json_data:
        labels = json.load(json_data)
    dataset_manager = DatasetManager(training_set,
                                     validation_set,
                                     test_set,
                                     genres,
                                     labels)

    batch_size = 1
    n_classes = 26


    # Graph input
    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_var = tf.placeholder(tf.float32)

    # Model
    pred = Model.alexnet(x, keep_var)  # definition of the network architecture

    # Loss and optimize

    # Init
    init = tf.global_variables_initializer()

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Load pretrained model
        # Skip weights from fc8 (fine-tuning)
        # load_with_skip('pretrained_alexnet.npy', sess, ['fc8'])
        saver.restore(sess, "saved_models/MSE_without_data_augmentation_0.75_0.001/film_genre_model.ckpt")
        print('Model Restored')

        '''test_map_global = 0.
        test_count = 0
        # test accuracy by group of batch_size images
        for _ in range(int(len(dataset_manager.test_list) / batch_size) +
                       1):
            batch_tx, batch_ty = dataset_manager.next_batch(
                batch_size, 'test')
            test_output = sess.run(pred,
                                   feed_dict={x: batch_tx,
                                              keep_var: 1})
            MAP = mean_average_precision(test_output, batch_ty)
            test_map_global += MAP
            test_count += 1
        test_map_global /= test_count
        print("Global Test Accuracy = {:.4f}".format(
            test_map_global))'''
        # Load one image
        img = cv2.imread(
            'saved_models/images_tests/naruto.jpg')
        img = cv2.resize(img, (227, 227))
        img = img.astype(np.float32)
        img -= np.array([104., 117., 124.])
        print(img)
        test_output = sess.run(pred,
                               feed_dict={x: np.reshape(img, (1, 227,227,3)),
                                          keep_var: 1})
        score_dict = {}
        for score, genre in zip(test_output[0], genres):
            score_dict[genre] = score
        print(list(reversed(sorted(score_dict.items(), key=lambda x:x[1]))))



if __name__ == '__main__':
    main()
