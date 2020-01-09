# from data import *
from tmp import *
from clstm import clstm_clf
import tensorflow as tf
import random
import numpy as np
import os
import sys
import time
import datetime


# TEST_VOCAB_FILENAME = TRAIN_FILENAME + '.vocab'

path1 = 'C:/20190530_final/KISVALUE_상폐/'
path2 = 'C:/20190530_final/KISVALUE_계속/'
def train():

    input = data_loader_X(path1) #상폐
    input2 = data_loader_O(path2) #계속상장
    # input.extend(input2)

    input_train_X = input[151:]
    input_test_X = input[:150]

    input_train_O = input2[201:]
    input_test_O = input2[:201]

    input_train_X.extend(input_train_O)
    input_test_X.extend(input_test_O)
    # print(len(input))

    with tf.Session() as sess:

        # seq_length = np.shape(input[0][0])[0]
        num_class = 2

        # print('initialize cnn filter')
        # print('number of class %d, vocab size %d' % (num_class, len(vocab)))
    # def __init__(self, max_length, num_classes, filter_sizes, num_filters, num_layer, l2_reg_lambda=0.0):
        cnn = clstm_clf(275, num_class, [3], 30, 2)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


         # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                # grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Write vocabulary
        # vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())



        def train_step(x_batch, y_batch, sequence_length):
            feed_dict = {

                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.sequence_length: sequence_length,
                cnn.keep_prob: 0.5
            }

            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, sequence_length, writer=None):
            feed_dict = {

                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.sequence_length: sequence_length,
                cnn.keep_prob: 1.0
            }

            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if step % 100 == 0:
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

            # print(len(box))
            # print(box)

        


        for i in range(5000):
            try:
                batch = random.sample(input_train_X, 64)

                x_batch, y_batch, sequence_length = zip(*batch)

                train_step(x_batch, y_batch, sequence_length)
                current_step = tf.train.global_step(sess, global_step)
            
                batch = random.sample(input_test_X, 64)
                x_test, y_test, sequence_length = zip(*batch)
                dev_step(x_test, y_test, sequence_length, writer=dev_summary_writer)
                if current_step % 1000 == 0:
                    save_path = saver.save(sess, './textcnn.ckpt')
                    print('model saved : %s' % save_path)

            except:
                print("Unexpected error:", sys.exc_info()[0])
                raise


if __name__ == '__main__':
    train()