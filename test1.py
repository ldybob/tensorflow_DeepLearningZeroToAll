"""Linear Regression1
입력 된 트레이닝 데이터로 학습"""
import tensorflow as tf
import matplotlib.pyplot as plt

x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []
for step in range(2001):
    cur_W, cur_cost, _ = sess.run([W, cost, train])
    if step % 200 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
    W_val.append(cur_W)
    cost_val.append(cur_cost)

plt.plot(W_val, cost_val)
plt.xlabel('W')
plt.ylabel('cost')
plt.show()

