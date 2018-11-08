"""
ML lab 12-3 Long Sequence RNN
"""

import tensorflow as tf
import numpy as np

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {c: i for i, c in enumerate(char_set)}

dic_size = char_set.__len__() # 입력 사이즈
num_classes = char_set.__len__() # 출력 사이즈
hidden_size = char_set.__len__()
sequence_length = 10
learning_late = 0.1

dataX = []
dataY = []
for i in range(0, sentence.__len__() - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + 1 + sequence_length]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    dataX.append(x)
    dataY.append(y)

batch_size = dataX.__len__()

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

X_one_hot = tf.one_hot(X, num_classes)

"""
num_units: 출력 사이즈
"""
def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    return cell

multi_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)
initial_state = multi_cells.zero_state(batch_size, tf.float32)

outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)

"""
위의 outputs은 rnn에 의해 activation function이 들어있었음.
그 값을 그대로 logits으로 전달하는 것은 잘못 된 것임.
그래서 아래와 같이 softmax layer를 깔고 logits 으로 전달하도록 함.
근데 아래는 softmax가 아닌 단순 fully connected layer인데???
"""
X_for_softmax = tf.reshape(outputs, [-1, hidden_size])
softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b", [num_classes])
outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_late).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(500):
        l, _ = sess.run([loss, train], feed_dict={X: dataX, Y: dataY})
        result = sess.run(prediction, feed_dict={X: dataX})
        print(i, 'loss: ', l, ' prediction: ', result, 'Y: ', dataY)
        #result_str = [char_set[c] for c in np.squeeze(result)]
        #print("\tPrediction str: ", ''.join(result_str))

    results = sess.run(prediction, feed_dict={X: dataX})
    for i, result in enumerate(results):
        if i == 0:
            print(''.join(char_set[c] for c in result), end='')
        else:
            print(char_set[result[-1]], end='')

