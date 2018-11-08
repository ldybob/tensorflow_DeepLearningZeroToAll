"""
ML lab 09-1
Wide and Deep NN for MNist
Add tensorboard
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, nb_classes])


W1 = tf.Variable(tf.random_normal([784, 256]), name='weight1')
b1 = tf.Variable(tf.random_normal([256]), name='bias1')
h1 = tf.nn.softmax(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, 256]), name='weight1')
b2 = tf.Variable(tf.random_normal([256]), name='bias1')
h2 = tf.nn.softmax(tf.matmul(h1, W2) + b2)

W3 = tf.Variable(tf.random_normal([256, nb_classes]), name='weight1')
b3 = tf.Variable(tf.random_normal([nb_classes]), name='bias1')

hypothesis = tf.nn.softmax(tf.matmul(h2, W3) + b3)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(cost)

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
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

    # tensor.eval은 session.run(tensor, ...)과 동일 함.
    #print("Accuracy: ", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print(r)
    print(mnist.test.labels[r:r+1])
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))
    print(mnist.test.images[r:r + 1])
    plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()

    W1_hist = tf.summary.histogram("weight1", W1)
    cost_summ = tf.summary.scalar("cost", cost)
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs')
    writer.add_graph(sess.graph)
    s, _ = sess.run([summary, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
    writer.add_summary(s, 100)



