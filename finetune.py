import tensorflow as tf
import pickle
import json
from datetime import datetime

from dataset_manager import DatasetManager
from model import Model
from network import load_with_skip
import sys


def main():
    # Load dataset manager
    with open('training_set_dict.pickle', 'rb') as handle:
        training_dict = pickle.load(handle)
    with open('test_set_dict.pickle', 'rb') as handle:
        test_dict = pickle.load(handle)
    with open('assets/genres.json') as json_data:
        genres = json.load(json_data)
    with open('assets/dataset.json') as json_data:
        dataset = json.load(json_data)
    dataset_manager = DatasetManager(training_dict,
                                     test_dict,
                                     genres,
                                     dataset)

    # Learning params
    learning_rate = 0.001
    batch_size = 50
    # Nombre d'iterations
    training_iters = 100
    # display training information (loss, training accuracy, ...) every 10
    # iterations
    display_step = 1
    test_step = 5  # test every test_step iterations

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
        tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(loss)

    # Evaluation
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
            batch_xs, batch_ys = dataset_manager.next_batch(
                batch_size, 'train'
            )
            sess.run(optimizer, feed_dict={
                     x: batch_xs, y: batch_ys, keep_var: keep_rate})

            #"""
            # Display testing status
            '''if step % test_step == 0:
                test_acc = 0.
                test_count = 0
                # test accuracy by group of batch_size images
                for _ in range(int(dataset.test_size / batch_size) + 1):
                    batch_tx, batch_ty = dataset.next_batch(
                        batch_size, 'test', loaded_img_test, loaded_lab_test)
                    print(batch_tx.shape)
                    acc = sess.run(accuracy, feed_dict={
                                   x: batch_tx, y: batch_ty, keep_var: 1.})
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print("{} Iter {}: Testing Accuracy = {:.4f}".format(
                    datetime.now(), step, test_acc))'''
            #"""
            # Display training status
            if step % display_step == 0:
                print(batch_xs.shape)
                print(batch_ys.shape)
                # Training-accuracy
                acc = sess.run(accuracy, feed_dict={
                               x: batch_xs, y: batch_ys, keep_var: 1.})
                batch_loss = sess.run(
                    loss, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})  # Training-loss
                print("{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}".format(
                    datetime.now(), step, batch_loss, acc))

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
