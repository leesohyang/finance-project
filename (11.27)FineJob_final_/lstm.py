import tensorflow as tf


class rnn_clf(object):
    """"

    LSTM and Bi-LSTM classifiers for text classification

    """

    # def __init__(self, max_length, num_classes, filter_sizes, num_filters, num_layer, l2_reg_lambda=0.0):

    def __init__(self, max_length, num_class, num_layer, l2_reg_lambda=0.0):

        # Placeholders
        # self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, max_length, 7], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='input_y')
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None, 1], name= 'sequence_length')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.num_layers = num_layer

        # L2 loss
        self.l2_loss = tf.constant(0.0)
        self.hidden_size = 50

        self.liist = []

        for i in range(1):
            self.liist.append(self.sequence_length[i][0])

        # Input dropout

        self.inputs = tf.nn.dropout(self.input_x, keep_prob=self.keep_prob)

        # LSTM
        def lstm_cell():
            cell = tf.contrib.rnn.LSTMCell(self.hidden_size,

                                           forget_bias=1.0,

                                           state_is_tuple=True,

                                           reuse=tf.get_variable_scope().reuse)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            return cell

        stackedRNNs = [lstm_cell() for _ in range(2)]
        multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True)
        # cell = tf.contrib.rnn.MultiRNNCell([cell] * self.num_layers, state_is_tuple=True)

        self._initial_state = multi_cells.zero_state(1, dtype=tf.float32)

        # Dynamic LSTM

        with tf.variable_scope('LSTM'):
            outputs, state = tf.nn.dynamic_rnn(multi_cells,

                                               inputs=self.inputs,

                                               initial_state=self._initial_state,

                                               sequence_length=self.liist)

            self.final_state = state


        # Softmax output layer

        with tf.name_scope('softmax'):

            softmax_w = tf.get_variable('softmax_w', shape=[self.hidden_size, 2], dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b', shape=[2], dtype=tf.float32)

            # L2 regularization for output layer

            self.l2_loss += tf.nn.l2_loss(softmax_w)

            self.l2_loss += tf.nn.l2_loss(softmax_b)

            # self.logits = tf.matmul(self.final_state[self.num_layers - 1].h, softmax_w) + softmax_b

            #softmax 전 완전연결층 
            self.logits = tf.nn.xw_plus_b(self.final_state[self.num_layers - 1].h, softmax_w, softmax_b, name='scores')
            self.hypothesis = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.logits, 1, name='predictions')
            # self.app_value = self.hypothesis[self.predictions]

            # self.predictions = tf.argmax(predictions, 1, name='predictions')

        # Loss

        with tf.name_scope('loss'):


            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2_loss


            # tvars = tf.trainable_variables()

            # # L2 regularization for LSTM weights

            # for tv in tvars:

            #     if 'kernel' in tv.name:
            #         self.l2_loss += tf.nn.l2_loss(tv)

            # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,

            #                                                         logits=self.logits)

            # self.cost = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        # Accuracy

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')


