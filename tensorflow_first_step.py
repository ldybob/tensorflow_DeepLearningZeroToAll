import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns

#텐서플로 첫걸음 - 선형회귀 분석 예제
"""
x_data = []
y_data = []
for i in range(1000):
    x_data.append(np.random.normal(0.0, 0.55))
    y_data.append(x_data[i] * 0.1 + 0.3 + np.random.normal(0.0, 0.03))

W = tf.Variable(tf.random_normal([1]), tf.float32)
b = tf.Variable(tf.random_normal([1]), tf.float32)
y = x_data * W + b

cost = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.52)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

fig = plt.figure()
for i in range(10):
    sess.run(train)
    ax = fig.add_subplot(5, 2, i + 1)
    ax.plot(x_data, y_data, 'o')
    ax.plot(x_data, sess.run(y))
    print(sess.run(cost))

plt.show()
"""

#==============================================================================


# 텐서플로 첫걸음 - k-mean 예제
num_points = 2000
vector_set = []
for i in range(num_points):
    if np.random.normal() > 0.5:
        vector_set.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
    else:
        vector_set.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

df = pd.DataFrame({"x": [v[0] for v in vector_set], "y": [v[1] for v in vector_set]})

#sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
#plt.show()

vectors = tf.constant(vector_set)
k = 4
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k, -1]))

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2), 0)

means = tf.concat([tf.reduce_mean(tf.gather(vectors,
                                tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])),
                                reduction_indices=[1]) for c in range(k)], 0)

update_centroids = tf.assign(centroids, means)

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

for step in range(100):
    _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

data = {"x": [], "y": [], "cluster": []}

for i in range(len(assignment_values)):
    data["x"].append(vector_set[i][0])
    data["y"].append(vector_set[i][1])
    data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data = df, fit_reg=False, size=6, hue="cluster", legend=False)

plt.show()

