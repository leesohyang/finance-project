from data_new2 import *
from att_lstm import AttLSTM
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
from PIL import Image
import random
VOCAB_FILENAME = 'vocab.vocab'
vocab = load_vocab(VOCAB_FILENAME)
# name = 입력받는 문장
def test(name):
    with tf.Session() as sess:
        # vocab = load_vocab(TRAIN_VOCAB_FILENAME)
        num_class = 3
        # vocab만 가져다 쓰기
        # input_x, total_label, vocab = data_loader_raw()

        cnn = AttLSTM(50, num_class, len(vocab), 128, 100)
        saver = tf.train.Saver()
        saver.restore(sess, './textcnn3_last.ckpt')
        print('model restored')

        
        # input_text = input('사용자 평가를 문장으로 입력하세요: ')

        input_tmp = data_loader_for_test(name)
        print(input_tmp)
        input_real = data_loader_last_for_test(input_tmp, vocab) #중간에 label도 들어가야함.

        data, length = zip(*input_real)
        print(data)

        feed_dict = {
            cnn.input_x: data,
            # cnn.input_y: y_batch,
            cnn.emb_dropout_keep_prob: 0.5,
            cnn.rnn_dropout_keep_prob: 0.5,
            cnn.dropout_keep_prob: 0.5
        }

        batch_accuracy = sess.run([cnn.hypothesis], feed_dict)
        # new_list = max(batch_accuracy)
        tmp_list = batch_accuracy[0]
        max_value = np.argmax(tmp_list)
        # sum_accuracy += batch_accuracy[0]
        # all_predictions = np.concatenate([all_predictions, batch_predictions])

        # final_accuracy = sum_accuracy / epoch_length
        print('Label: {}'.format(max_value))
        return max_value

if __name__ == '__main__':
    test()