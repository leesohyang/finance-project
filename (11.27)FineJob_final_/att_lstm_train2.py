# from data import *
from data_new2 import *
from att_lstm import AttLSTM
import tensorflow as tf
import random
import numpy as np
import os
import sys
import time
import datetime
# tensorboard –logdir=./logs
#
# http://localhost:6006
VOCAB_FILENAME = 'vocab.vocab'
tensorboard_path = './tensorboard'

print("loading input")
input_x, total_label, vocab = data_loader_raw()
save_vocab(VOCAB_FILENAME, vocab)
input = data_loader_last(input_x, total_label, vocab)
input_train = input[:32000]
input_test = input[32000:]
# label_train =
print("loading input complete")

with tf.Graph().as_default():
    sess = tf.Session()
    with sess:

        num_class = 3

        cnn = AttLSTM(50, num_class, len(vocab), 128, 100)

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
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.emb_dropout_keep_prob: 0.5,
                cnn.rnn_dropout_keep_prob: 0.5,
                cnn.dropout_keep_prob: 0.5
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, y_batch, sequence_length, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.emb_dropout_keep_prob: 1.0,
                cnn.rnn_dropout_keep_prob: 1.0,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if step % 100 == 0:
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # -----여기서부터 수정

        for i in range(10000):
            try:
                batch = random.sample(input_train, 32)

                x_batch, y_batch, sequence_length = zip(*batch)

                train_step(x_batch, y_batch, sequence_length)
                current_step = tf.train.global_step(sess, global_step)
                # if current_step % 100 == 0:
                batch = random.sample(input_test, 32)
                x_test, y_test, sequence_length = zip(*batch)
                dev_step(x_test, y_test, sequence_length, writer=dev_summary_writer)
                if current_step % 1000 == 0:
                    save_path = saver.save(sess, './textcnn3_last.ckpt')
                    print('model saved : %s' % save_path)


            except:
                print("Unexpected error:", sys.exc_info()[0])
                raise
        # -----------------------
        # Generate batches
        # batches = data_helpers.batch_iter(
        #     list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # # Training loop. For each batch...
        # for batch in batches:
        #     x_batch, y_batch = zip(*batch)
        #     train_step(x_batch, y_batch)
        #     current_step = tf.train.global_step(sess, global_step)
        #     if current_step % FLAGS.evaluate_every == 0:
        #         print("\nEvaluation:")
                # dev_step(x_dev, y_dev, writer=dev_summary_writer)
        #         print("")
        #     if current_step % FLAGS.checkpoint_every == 0:
        #         path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        #         print("Saved model checkpoint to {}\n".format(path))


# ---------------------------원래꺼
# def train():
#     print("loading input")
#     input_x, total_label, vocab = data_loader_raw()
#     input = data_loader_last(input_x, total_label, vocab)
#     input_train = input[:70]
#     input_test = input[70:100]
#     # label_train =
#     print("loading input complete")
#
#     with tf.Session() as sess:
#
#         # seq_length = np.shape(input[0][0])[0]
#         num_class = 5
#
#         cnn = AttLSTM(50, num_class, len(vocab), 128, 100)
#
#         global_step = tf.Variable(0, name='global_step', trainable=False)
#         optimizer = tf.train.AdamOptimizer(1e-3)
#         grads_and_vars = optimizer.compute_gradients(cnn.loss)
#         train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
#
#         def train_step(x_batch, y_batch, sequence_length):
#             feed_dict = {
#
#                 # cnn.sequence_length: sequence_length,
#                 cnn.input_x: x_batch,
#                 cnn.input_y: y_batch,
#                 cnn.emb_dropout_keep_prob: 0.5,
#                 cnn.rnn_dropout_keep_prob: 0.5,
#                 cnn.dropout_keep_prob: 0.5
#             }
#
#             _, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
#
#         def evaluate(x_batch, y_batch, sequence_length):
#             feed_dict = {
#
#                 # cnn.sequence_length: sequence_length,
#                 cnn.input_x: x_batch,
#                 cnn.input_y: y_batch,
#                 cnn.emb_dropout_keep_prob: 1.0,
#                 cnn.rnn_dropout_keep_prob: 1.0,
#                 cnn.dropout_keep_prob: 1.0
#             }
#
#             step, loss, accuracy = sess.run([global_step, cnn.loss, cnn.accuracy], feed_dict)
#
#             print("step %d, loss %f, acc %f" % (step, loss, accuracy))
#             # print(len(box))
#             # print(box)
#
#         saver = tf.train.Saver()
#         sess.run(tf.global_variables_initializer())
#
#         for i in range(10):
#             try:
#                 batch = random.sample(input_train, 3)
#
#                 x_batch, y_batch, sequence_length = zip(*batch)
#
#                 train_step(x_batch, y_batch, sequence_length)
#                 current_step = tf.train.global_step(sess, global_step)
#                 if current_step % 10 == 0:
#                     batch = random.sample(input_test, 3)
#                     x_test, y_test, sequence_length = zip(*batch)
#                     evaluate(x_test, y_test, sequence_length)
#                 if current_step % 10 == 0:
#                     save_path = saver.save(sess, './textcnn.ckpt')
#                     print('model saved : %s' % save_path)
#
#
#             except:
#                 print("Unexpected error:", sys.exc_info()[0])
#                 raise
#
#
# if __name__ == '__main__':
#     train()