"""텐서플로에서 지원하는 Softmax cross entropy를 사용한 학습 구현"""

import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7

X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.int32, shape=[None, 1])

# Y 데이터는 0~6 까지의 값이기 때문에 one_hot(0:[1000000], 1:[0100000].....) 방식으로 바꾸기 위한 코드
Y_one_hot = tf.one_hot(Y, nb_classes)
# one_hot 함수 사용 시 차원이 증가하게 되어 차원 감소를 위해 reshape 사용
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='weight')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

"""아래 두 줄의 cost 구하는 코드는
 test12.py 의 cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))를 텐서플로에서 지원하는 함수를 사용하도록 한 것임."""
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    pred = sess.run(prediction, feed_dict={X: x_data})
    for p, y in zip(pred, y_data.flatten()):
        print(p == int(y), p, int(y))
