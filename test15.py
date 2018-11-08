"""
XOR with ReLU
9 hidden layer
"""

import tensorflow as tf
import numpy as np

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 5], -1.0, 1.0), name='weight1')
b1 = tf.Variable(tf.zeros([5]), name='bias1')
with tf.name_scope("layer1") as scope:
    layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight2')
b2 = tf.Variable(tf.zeros([5]), name='bias2')
with tf.name_scope("layer2") as scope:
    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight3')
b3 = tf.Variable(tf.zeros([5]), name='bias3')
with tf.name_scope("layer3") as scope:
    layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight4')
b4 = tf.Variable(tf.zeros([5]), name='bias4')
with tf.name_scope("layer4") as scope:
    layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)

W5 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight5')
b5 = tf.Variable(tf.zeros([5]), name='bias5')
with tf.name_scope("layer5") as scope:
    layer5 = tf.nn.relu(tf.matmul(layer4, W5) + b5)

W6 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight6')
b6 = tf.Variable(tf.zeros([5]), name='bias6')
with tf.name_scope("layer6") as scope:
    layer6 = tf.nn.relu(tf.matmul(layer5, W6) + b6)

W7 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight7')
b7 = tf.Variable(tf.zeros([5]), name='bias7')
with tf.name_scope("layer7") as scope:
    layer7 = tf.nn.relu(tf.matmul(layer6, W7) + b7)

W8 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight8')
b8 = tf.Variable(tf.zeros([5]), name='bias8')
with tf.name_scope("layer8") as scope:
    layer8 = tf.nn.relu(tf.matmul(layer7, W8) + b8)

W9 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight9')
b9 = tf.Variable(tf.zeros([5]), name='bias9')
with tf.name_scope("layer9") as scope:
    layer9 = tf.nn.relu(tf.matmul(layer8, W9) + b9)

W10 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight0')
b10 = tf.Variable(tf.zeros([5]), name='bias10')
with tf.name_scope("layer10") as scope:
    layer10 = tf.nn.relu(tf.matmul(layer9, W10) + b10)

W11 = tf.Variable(tf.random_uniform([5, 1], -1.0, 1.0), name='weight11')
b11 = tf.Variable(tf.zeros([1]), name='bias11')
with tf.name_scope("layer11") as scope:
    hypothesis = tf.sigmoid(tf.matmul(layer10, W11) + b11)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print(h, c, a)

    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs')
    writer.add_graph(sess.graph)

