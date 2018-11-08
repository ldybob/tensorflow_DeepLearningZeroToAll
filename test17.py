"""
ML lab 12-2 RNN Hi Hello Training
"""

import numpy as np
import tensorflow as tf

idx2char = ['h', 'i', 'e', 'l', 'o']

h = [1, 0, 0, 0, 0]
i = [0, 1, 0, 0, 0]
e = [0, 0, 1, 0, 0]
l = [0, 0, 0, 1, 0]
o = [0, 0, 0, 0, 1]

x_data = [[0, 1, 0, 2, 3, 3]]
y_data = [[1, 0, 2, 3, 3, 4],
          [2, 3, 3, 4, 0, 1]]
x_one_hot = [[h, i, h, e, l, l],
             [h, e, l, l, o, h]]

num_classes = 5
input_dim = 5
hidden_size = 2
sequence_length = 6
batch_size = 2
learning_late = 0.1

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])
Y = tf.placeholder(tf.int32, [None, sequence_length])

"""
num_units: 출력 사이즈
"""
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)

"""
dynamic_rnn함수는 cell만 바꿔서 사용 할 수 있도록 구성 됨
"""
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)
weights = tf.ones([batch_size, sequence_length])

X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(
    inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_late).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2000):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, 'loss: ', l, ' prediction: ', result, 'Y: ', y_data)
        result_str = [idx2char[c] for c in np.squeeze(result.reshape(-1))]
        print("\tPrediction str: ", ''.join(result_str))
