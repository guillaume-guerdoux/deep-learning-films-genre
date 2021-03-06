import tensorflow as tf
import pickle
import json

from datetime import datetime
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

    log_file_name = str(datetime.now()) + '-logs.txt'
    with open("logs/" + log_file_name, 'w') as log_file:
        log_file.write('Training logs \n')

    iteration_file_name = str(datetime.now()) + '-iteration.txt'
    with open("logs/" + iteration_file_name, 'w') as log_file:
        log_file.write('Training iterations \n')

    best_iteration_file_name = str(datetime.now()) + '-best_iteration.txt'
    with open("logs/" + best_iteration_file_name, 'w') as log_file:
        log_file.write('Best Training iterations \n')

    dataset_manager = DatasetManager(training_set,
                                     validation_set,
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
    local_train_step = 20
    global_validation_step = 25  # test every global_test_step iterations
    global_train_step = 100

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
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, pred))))
    # optimizer = tf.train.GradientDescentOptimizer(
    #    learning_rate=learning_rate).minimize(loss)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

    # Init
    init = tf.global_variables_initializer()

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # To do early stopping
    max_validation_map = 0

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Load pretrained model
        # Skip weights from fc8 (fine-tuning)
        load_with_skip('pretrained_alexnet.npy', sess, ['fc8'])

        print('Start training.')
        step = 1
        while step < training_iters:
            # print("Iter ", step)
            with open("logs/" + iteration_file_name, 'a') as log_file:
                log_file.write("Iter {} \n".format(
                    step))
            batch_xs, batch_ys = dataset_manager.next_batch(
                batch_size, 'train'
            )
            sess.run(optimizer, feed_dict={
                     x: batch_xs, y: batch_ys, keep_var: keep_rate})

            # Display on batch training status
            if step % local_train_step == 0:
                local_train_output = sess.run(
                    pred, feed_dict={x: batch_xs, keep_var: 1})
                MAP = mean_average_precision(local_train_output, batch_ys)
                batch_loss = sess.run(
                    loss, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
                with open("logs/" + log_file_name, 'a') as log_file:
                    log_file.write("Iter {} Training Loss = {:.4f}, "
                                   "Mean average precision = {:.4f} \n".format(
                                    step, batch_loss, MAP))

            # Display global training error
            if step % global_train_step == 0:
                train_map_global = 0.
                test_count = 0
                # test accuracy by group of batch_size images
                for _ in range(int(len(dataset_manager.training_list) / batch_size) + 1):
                    batch_tx, batch_ty = dataset_manager.next_batch(
                        batch_size, 'train')
                    test_output = sess.run(
                        pred, feed_dict={x: batch_tx, keep_var: 1})
                    MAP = mean_average_precision(test_output, batch_ty)
                    train_map_global += MAP
                    test_count += 1
                train_map_global /= test_count
                with open("logs/" + log_file_name, 'a') as log_file:
                    log_file.write("Global Training Accuracy = {:.4f} \n".format(
                        train_map_global))

            # Display global testing error
            if step % global_validation_step == 0:
                validation_map_global = 0.
                validation_count = 0
                # test accuracy by group of batch_size images
                for _ in range(int(len(dataset_manager.test_list) / batch_size) +
                               1):
                    batch_tx, batch_ty = dataset_manager.next_batch(
                        batch_size, 'val')
                    test_output = sess.run(pred,
                                           feed_dict={x: batch_tx,
                                                      keep_var: 1})
                    MAP = mean_average_precision(test_output, batch_ty)
                    validation_map_global += MAP
                    validation_count += 1
                validation_map_global /= validation_count
                with open("logs/" + log_file_name, 'a') as log_file:
                    log_file.write("Iter {} Global Validation Accuracy = {:.4f} \n".format(
                        step, validation_map_global))
                if validation_map_global >= max_validation_map:
                    max_validation_map = validation_map_global
                    with open("logs/" + best_iteration_file_name, 'a') as log_file:
                        log_file.write("Iter {}  \n".format(
                                        step))
                    # Save model
                    saver.save(sess, "saved_models/model_dropout05_mean_square_error.ckpt")

            step += 1
        # print("Finish!")
        with open("logs/finish", 'w') as finish_file:
            finish_file.write("Finish")



if __name__ == '__main__':
    main()
