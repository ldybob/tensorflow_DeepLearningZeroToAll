"""
ML lab 12-3 Long Sequence RNN
"""

import tensorflow as tf
import numpy as np

sample = "if you want you"
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)}

sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]
print(np.array(x_data).shape, np.array(y_data).shape)

dic_size = idx2char.__len__() # 입력 사이즈
num_classes = idx2char.__len__() # 출력    사이즈
hidden_size = idx2char.__len__()
batch_size = 1
sequence_length = sample.__len__() - 1
learning_late = 0.1

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

X_one_hot = tf.one_hot(X, num_classes)

"""
num_units: 출력 사이즈
"""
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)

outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)
weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_late).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})
        print(i, 'loss: ', l, ' prediction: ', result, 'Y: ', y_data)
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))
