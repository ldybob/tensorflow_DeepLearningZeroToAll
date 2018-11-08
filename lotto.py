import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


class Lotto:
    def __init__(self):
        self.lotto_number = []
        self.sample = [i + 1 for i in range(45)]
        self.value2idx = {c: i for i, c in enumerate(self.sample)}
        self.xy = np.loadtxt('lotto.csv', delimiter=',')
        self.xy = self.xy.reshape([-1, 6])
        self.xy = self.xy[::-1]
        self.data = []
        # for i in range(6):
        #     self.data.append(self.xy[:, i])
        self.data = self.xy.T
        self.train_size = int(len(self.data[0]) * 0.8)

        self.input_dim = self.sample.__len__()  # 입력 사이즈
        self.output_dim = self.sample.__len__()  # 출력 사이즈
        self.hidden_size = self.sample.__len__()
        self.sequence_length = 20
        self.learning_rate = 0.05
        self.batch_size = 1

    def check_duplicate(self, numbers, number):
        duplicate = False
        for n in numbers:
            if n == number:
                duplicate = True
        return duplicate

    def train(self):
        lotto_digits = 0
        namespace = 0
        while lotto_digits < 6:
            namespace += 1
            with tf.variable_scope('digits' + str(namespace)):
                dataX = []
                dataY = []
                for i in range(0, self.data[lotto_digits].__len__() - self.sequence_length):
                    x_str = self.data[lotto_digits][i:i + self.sequence_length]
                    y_str = self.data[lotto_digits][i + 1: i + 1 + self.sequence_length]
                    # print(i, x_str, '->', y_str)

                    x = [self.value2idx[c] for c in x_str]
                    y = [self.value2idx[c] for c in y_str]

                    dataX.append(x)
                    dataY.append(y)

                self.batch_size = dataX.__len__()

                X = tf.placeholder(tf.int64, [None, self.sequence_length])
                Y = tf.placeholder(tf.int64, [None, self.sequence_length])

                X_one_hot = tf.one_hot(X, self.output_dim)

                """
                num_units: RNN 출력 사이즈
                뒤에 fully connected가 있기 때문에 num_units사이즈 임의로 정함.
                """
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size, state_is_tuple=True)

                multi_cells = tf.contrib.rnn.MultiRNNCell([cell for _ in range(2)], state_is_tuple=True)
                initial_state = multi_cells.zero_state(self.batch_size, tf.float32)

                # outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)
                outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, initial_state=initial_state, dtype=tf.float32)

                """
                위의 outputs은 rnn에 의해 activation function이 들어있었음.
                그 값을 그대로 logits으로 전달하는 것은 잘못 된 것임.
                그래서 아래와 같이 softmax layer를 깔고 logits 으로 전달하도록 함.
                근데 아래는 softmax가 아닌 단순 fully connected layer인데???
                """
                X_for_softmax = tf.reshape(outputs, [-1, self.hidden_size])
                softmax_w = tf.get_variable("softmax_w", [self.hidden_size, self.output_dim])
                softmax_b = tf.get_variable("softmax_b", [self.output_dim])
                outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b
                outputs = tf.reshape(outputs, [self.batch_size, self.sequence_length, self.output_dim])

                weights = tf.ones([self.batch_size, self.sequence_length])

                sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
                loss = tf.reduce_mean(sequence_loss)
                train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

                prediction = tf.argmax(outputs, axis=2)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), dtype=tf.float32))

                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())

                    for i in range(500):
                        l, _ = sess.run([loss, train], feed_dict={X: dataX, Y: dataY})
                        print(i, 'loss: ', l)

                    results = sess.run(prediction, feed_dict={X: dataY})
                    print(self.sample[results[-1][-1]])
                    if self.check_duplicate(self.lotto_number, self.sample[results[-1][-1]]):
                        continue
                    else:
                        self.lotto_number.append(self.sample[results[-1][-1]])
                    results, acc = sess.run([prediction, accuracy], feed_dict={X: dataX, Y: dataY})
                    # for i, result in enumerate(results):
                    #     if i == 0:
                    #         for v in result:
                    #             print(sample[v])
                    #         # print(''.join(str(sample[v] for v in result)), end='')
                    #     else:
                    #         print(sample[result[-1]])
                    print('accuracy : ', acc)
                lotto_digits += 1
                if lotto_digits == 1:
                    graph_X, graph_Y = None, None

                    for i, result in enumerate(dataY):
                        if i == 0:
                            graph_X = result
                        else:
                            graph_X.append(result[-1])

                    for i, result in enumerate(results):
                        if i == 0:
                            graph_Y = result.tolist()
                        else:
                            graph_Y.append(result[-1])
                    fig = plt.figure()
                    ax1 = fig.add_subplot(2, 1, 1)
                    ax1.plot(graph_X)
                    ax2 = fig.add_subplot(2, 1, 2)
                    ax2.plot(graph_Y)
                    plt.show()


    def get_lotto_number(self):
        self.lotto_number.sort()
        return self.lotto_number

if __name__ == "__main__":
    lotto = Lotto()
    lotto.train()
    print(lotto.get_lotto_number())
