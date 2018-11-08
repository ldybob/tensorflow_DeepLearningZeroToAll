"""
ML lab 05 Exercise
- CSV reading using tf.decode_csv
- Try other classification data from Kaggle
"""

import tensorflow as tf
import numpy as np

# normalization
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


sess = tf.Session()

filename_queue = tf.train.string_input_producer(['creditcard.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[0.]] * 31
data = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = tf.train.batch([data[1:-2], data[-1:]], batch_size=10)

X = tf.placeholder(tf.float32, shape=[None, 28])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([28, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(5000):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    x_batch = MinMaxScaler(x_batch)

    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, cost_val)

h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_batch, Y: y_batch})
print(h, c, a)

coord.request_stop()
coord.join(threads)