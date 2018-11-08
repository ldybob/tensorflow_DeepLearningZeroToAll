"""
ML lab 11-3
CNN using class, Layers, Ensemble
"""

import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

class Model:
    """
    img_x : 이미지의 가로 픽셀 사이즈
    img_y : 이미지의 세로 픽셀 사이즈
    label : 정답 label의 사이즈
    """
    def __init__(self, name, session, img_x, img_y, label):
        self.name = name
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
        with tf.variable_scope(self.name):
            conv1 = tf.layers.conv2d(inputs=self.X_img, filters=32, kernel_size=[3, 3], padding="SAME",
                                     activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.3, training=self.training)
            # conv -> (?, 28, 28, 32)
            # pool -> (?, 14, 14, 32)

            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding="SAME",
                                     activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2, rate=0.3, training=self.training)
            # conv -> (?, 14, 14, 64)
            # pool -> (?, 7, 7, 64)

            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], padding="SAME",
                                     activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="SAME", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.3, training=self.training)
            # conv -> (?, 7, 7, 128)
            # pool -> (?, 4, 4, 128)
            flat = tf.reshape(dropout3, [-1, 4 * 4 * 128])

            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            dropout4 = tf.layers.dropout(inputs=dense4, rate=0.3, training=self.training)

            W5 = tf.get_variable('W5', [625, 10], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.random_normal([10]))

            self.hypothesis = tf.matmul(dropout4, W5) + b

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

    def predict(self, x_data):
        return sess.run(self.hypothesis, feed_dict={self.X: x_data,  self.training: False})

    def get_predicted_number(self, x_data):
        return sess.run(tf.argmax(self.hypothesis, 1), feed_dict={self.X: x_data, self.training: False})


if __name__ == '__main__':
    print('main')
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    sess = tf.Session()
    models = []
    num_models = 3
    for m in range(num_models):
        models.append(Model('model' + str(m), sess, 28, 28, 10))

    sess.run(tf.global_variables_initializer())

    training_epochs = 1
    batch_size = 100

    for epoch in range(training_epochs):
        avg_cost_list = np.zeros(num_models)
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            for m_idx, m in enumerate(models):
                c, _ = models[m_idx].train(batch_xs, batch_ys, 0.7)
                avg_cost_list[m_idx] += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost = ', avg_cost_list)

    # 각 모델별 예측 숫자 값 출력
    # 원하는 건 각각 모델들의 예측 labael의 합을 구해 최종 예측 숫자 출력
    r = random.randint(0, mnist.test.num_examples - 1)
    for m in models:
        print(m.get_predicted_number(mnist.test.images[r:r + 1]))
    plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()

    # 모델별 예측 label의 합을 구하여 예측 숫자 구한 후 평균 accuracy를 구함
    # 노트북에서는 OOM 발생
    predictions = 0
    for i, m in enumerate(models):
        print('acc : ' + str(i), m.evaluate(mnist.test.images, mnist.test.labels))
        p = m.predict(mnist.test.images)
        predictions += p

    ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
    ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))

    print('ensemble_accuracy : ', sess.run(ensemble_accuracy))

