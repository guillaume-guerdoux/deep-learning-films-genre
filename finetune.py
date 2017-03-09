import tensorflow as tf
import pickle
import json
from datetime import datetime

from dataset_manager import DatasetManager
from model import Model
from network import load_with_skip
from network import mean_average_precision
import sys
import time


def main():
    # Load dataset manager
    with open('training_set_list.pickle', 'rb') as handle:
        training_set = pickle.load(handle)
    with open('test_set_list.pickle', 'rb') as handle:
        test_set = pickle.load(handle)
    with open('genres.json') as json_data:
        genres = json.load(json_data)
    with open('labels.json') as json_data:
        labels = json.load(json_data)
    dataset_manager = DatasetManager(training_set,
                                     test_set,
                                     genres,
                                     labels)

    # Learning params
    learning_rate = 0.001
    batch_size = 50
    # Nombre d'iterations
    training_iters = 1000
    # display training information (loss, training accuracy, ...) every 10
    # iterations
    local_train_step = 10
    global_test_step = 50  # test every global_test_step iterations
    global_train_step = 75

    # Network params
    n_classes = 26
    keep_rate = 0.5  # for dropout

    # Graph input
    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_var = tf.placeholder(tf.float32)

    # Model
    pred = Model.alexnet(x, keep_var)  # definition of the network architecture

    # Loss and optimizer
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(loss)

    # Init
    init = tf.global_variables_initializer()

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Load pretrained model
        # Skip weights from fc8 (fine-tuning)
        load_with_skip('pretrained_alexnet.npy', sess, ['fc8'])

        print('Start training.')
        step = 1
        while step < training_iters:
            print("Iter {}".format(step))
            batch_xs, batch_ys = dataset_manager.next_batch(
                batch_size, 'train'
            )
            sess.run(optimizer, feed_dict={
                     x: batch_xs, y: batch_ys, keep_var: keep_rate})

            # Display global testing error
            if step % global_test_step == 0:
                test_map_global = 0.
                test_count = 0
                # test accuracy by group of batch_size images
                for _ in range(int(len(dataset_manager.test_list)/batch_size) +
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
                print("Global Testing Accuracy = {:.4f}".format(
                       test_map_global))

            # Display global training error
            if step % global_train_step == 0:
                train_map_global = 0.
                test_count = 0
                # test accuracy by group of batch_size images
                for _ in range(int(len(dataset_manager.training_list) / batch_size) + 1):
                    batch_tx, batch_ty = dataset_manager.next_batch(
                        batch_size, 'train')
                    test_output = sess.run(pred, feed_dict={x: batch_tx, keep_var: 1})
                    MAP = mean_average_precision(test_output, batch_ty)
                    train_map_global += MAP
                    test_count += 1
                train_map_global /= test_count
                print("Global Training Accuracy = {:.4f}".format(
                       train_map_global))

            # Display on batch training status
            if step % local_train_step == 0:
                test_output = sess.run(pred, feed_dict={x: batch_xs, keep_var: 1})
                MAP = mean_average_precision(test_output, batch_ys)
                batch_loss = sess.run(
                    loss, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})  # Training-loss
                print("Training Loss = {:.4f}, Mean average precision = {:.4f}".format(
                      batch_loss, MAP))

            step += 1
        print("Finish!")
        # Save model weights to disk
        # save_path = saver.save(sess, "model.ckpt")
        # x_test = tf.placeholder(tf.float32, [1, 227, 227, 3])
        # y_test = tf.placeholder(tf.float32, [None, n_classes])
        #save checkpoint of the model
        # save_path = saver.save(sess, "saved_models/test_model.ckpt")

        '''img = loaded_img_train[0][:][:][:]
        label = loaded_lab_train[0][:]
        print(img.shape)
        print(label.shape)
        acc = sess.run(accuracy, feed_dict={
                       x_test: [img], y_test: [label], keep_var: 1.})
        print(acc)
        one_image = loaded_img_train[0][:][:][:]
        prediction = sess.run(pred, feed_dict={x: [img], y: [label], keep_var: 1.})
        print(prediction)'''

if __name__ == '__main__':
    main()
