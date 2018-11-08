"""
ML lab 11-3
CNN using class
CNN using Layers
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Model:
    """
    img_x : 이미지의 가로 픽셀 사이즈
    img_y : 이미지의 세로 픽셀 사이즈
    label : 정답 label의 사이즈
    """
    def __init__(self, session, img_x, img_y, label):
        self.sess = session
        self.X = tf.placeholder(tf.float32, [None, img_x * img_y])
        self.X_img = tf.reshape(self.X, [-1, img_x, img_y, 1])
        self.Y = tf.placeholder(tf.float32, [None, label])
        self.keep_prob = tf.placeholder(tf.float32)
        self.training = tf.placeholder(tf.bool)
        self.build_net()

        self.is_correct = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.is_correct, tf.float32))


    def build_net(self):
        """
        W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
        L1 = tf.nn.conv2d(self.X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)
        """
        conv1 = tf.layers.conv2d(inputs=self.X_img, filters=32, kernel_size=[3, 3], padding="SAME",
                                 activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)
        dropout1 = tf.layers.dropout(inputs=pool1, rate=0.3, training=self.training)
        # conv -> (?, 28, 28, 32)
        # pool -> (?, 14, 14, 32)

        """
        W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
        """
        conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding="SAME",
                                 activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)
        dropout2 = tf.layers.dropout(inputs=pool2, rate=0.3, training=self.training)
        # conv -> (?, 14, 14, 64)
        # pool -> (?, 7, 7, 64)

        """
        W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
        L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
        L3 = tf.nn.relu(L3)
        L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
        L3 = tf.reshape(L3, [-1, 4 * 4 * 128])
        """
        conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], padding="SAME",
                                 activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="SAME", strides=2)
        dropout3 = tf.layers.dropout(inputs=pool3, rate=0.3, training=self.training)
        # conv -> (?, 7, 7, 128)
        # pool -> (?, 4, 4, 128)
        flat = tf.reshape(dropout3, [-1, 4 * 4 * 128])

        """
        W4 = tf.get_variable('W4', [4 * 4 * 128, 625], initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.Variable(tf.random_normal([625]))
        L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
        L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)
        """
        dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        dropout4 = tf.layers.dropout(inputs=dense4, rate=0.3, training=self.training)

        W5 = tf.get_variable('W5', [625, 10], initializer=tf.contrib.layers.xavier_initializer())
        b5 = tf.Variable(tf.random_normal([10]))

        self.hypothesis = tf.matmul(dropout4, W5) + b5

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.hypothesis, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)

    def train(self, x_data, y_data, keep_prob=0.7):
        return sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data,
                                                                self.keep_prob: keep_prob, self.training: True})

    def evaluate(self, x_data, y_data, batch_size=512):
        N = x_data.shape[0]
        acc = 0

        for i in range(0, N, batch_size):
            batch_x = x_data[i: i + batch_size]
            batch_y = y_data[i: i + batch_size]
            N_batch = batch_x.shape[0]

            acc += sess.run(self.accuracy, feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 1, self.training: False}) * N_batch

        return acc / N

    def get_accuracy(self, x_data, y_data):
        return self.evaluate(x_data, y_data)

if __name__ == '__main__':
    print('main')
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    sess = tf.Session()
    model = Model(sess, 28, 28, 10)

    sess.run(tf.global_variables_initializer())

    training_epochs = 10
    batch_size = 100

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = model.train(batch_xs, batch_ys, 0.7)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

    print("Accuracy: ", model.evaluate(mnist.test.images, mnist.test.labels))

