from tmp import *
from clstm import clstm_clf
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
from PIL import Image
import random

SEQUENCE_LENGTH = 60
NUM_CLASS = 2

path = 'C:/20190530_final/data_sample/' #SKC코오롱PI -코스닥 178920.xlsx


def test_finanace(company):
    with tf.Session() as sess:
        dataIdx_dic = {"이녹스(주)": "088390"}
        company_code = dataIdx_dic[company]
        # path = 'C:/20190530_final/data_sample/' + company_code + '.xls'
        # vocab = load_vocab(TRAIN_VOCAB_FILENAME)
        input = data_loader_O_for_test(company_code)
        cnn = clstm_clf(275, 2, [3], 30, 2)
        saver = tf.train.Saver()
        saver.restore(sess, './textcnn.ckpt')
        print('model restored')

        # input_text = input('사용자 평가를 문장으로 입력하세요: ')
        # input_text = load_data(TEST_DATA_FILENAME)

        # all_predictions = []
        # sum_accuracy = 0
        #
        # epoch_length = len(input_text) // 100  # 배치를 몇번 할수 있느냐
        # # x_batch, y_batch = zip(*bat
        #
        # for j in range(epoch_length):
        #     input = []
        #     start_index = j * 100
        #     end_index = start_index + 100
        #     input = input_text[start_index:end_index][:]
        data, label, length = zip(*input)

        feed_dict = {
            cnn.input_x: data,
            cnn.input_y: label,
            cnn.sequence_length: length,
            cnn.keep_prob: 1.0
        }

        batch_accuracy = sess.run([cnn.hypothesis], feed_dict)

        new_list = max(batch_accuracy)
        max_value = new_list[0][0]
            # sum_accuracy += batch_accuracy[0]
            # all_predictions = np.concatenate([all_predictions, batch_predictions])

        # final_accuracy = sum_accuracy / epoch_length
        print('Test accuracy: {}'.format(max_value))
        return max_value


if __name__ == '__main__':
    test_finanace()