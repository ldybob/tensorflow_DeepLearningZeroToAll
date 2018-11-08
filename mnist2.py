"""
ML lab 10
NN, ReLU, Xavier, Adam
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, nb_classes])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable("W1", shape=[784,512], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]), name='bias1')
h1 = tf.nn.relu(tf.matmul(X, W1) + b1)
h1 = tf.nn.dropout(h1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[512,512], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]), name='bias2')
h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
h2 = tf.nn.dropout(h2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[512,512], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]), name='bias3')
h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
h3 = tf.nn.dropout(h3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[512,512], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]), name='bias4')
h4 = tf.nn.relu(tf.matmul(h3, W4) + b4)
h4 = tf.nn.dropout(h4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[512,nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([nb_classes]), name='bias5')
hypothesis = tf.matmul(h4, W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

    # tensor.eval은 session.run(tensor, ...)과 동일 함.
    #print("Accuracy: ", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print(r)
    print(mnist.test.labels[r:r+1])
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))
    print(mnist.test.images[r:r + 1])
    plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()


