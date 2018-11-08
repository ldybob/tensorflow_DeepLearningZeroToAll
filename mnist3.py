"""
ML lab 11-2 MNIST 99% with CNN
"""

import tensorflow as tf
import datetime
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
# batch_normalization 을 사용하는 경우 아래 dropout 필요없음
L1 = tf.layers.batch_normalization(L1)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
# conv -> (?, 28, 28, 32)
# pool -> (?, 14, 14, 32)

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
# batch_normalization 을 사용하는 경우 아래 dropout 필요없음
L2 = tf.layers.batch_normalization(L2)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
# conv -> (?, 14, 14, 64)
# pool -> (?, 7, 7, 64)

W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
# batch_normalization 을 사용하는 경우 아래 dropout 필요없음
L3 = tf.layers.batch_normalization(L3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3 = tf.reshape(L3, [-1, 4 * 4 * 128])
# conv -> (?, 7, 7, 128)
# pool -> (?, 4, 4, 128)

W4 = tf.get_variable('W4', [4 * 4 * 128, 625], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
# batch_normalization 을 사용하는 경우 아래 dropout 필요없음
L4 = tf.layers.batch_normalization(L3)
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
# L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable('W5', [625, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 3
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('Learning start. ', datetime.datetime.now())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
    print('Learning End. ', datetime.datetime.now())

    #print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

    def evaluate(x_data, y_data, batch_size=512):
        N = x_data.shape[0]
        acc = 0

        for i in range(0, N, batch_size):
            batch_x = x_data[i: i + batch_size]
            batch_y = y_data[i: i + batch_size]
            N_batch = batch_x.shape[0]

            acc += sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1}) * N_batch

        return acc / N

    print("Accuracy: ", evaluate(mnist.test.images, mnist.test.labels))
    sess.close()

