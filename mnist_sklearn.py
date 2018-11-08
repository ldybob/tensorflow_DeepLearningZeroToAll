"""
MNIST classification use sklearn svm
"""
from sklearn import svm,metrics
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

clf = svm.SVC()
clf.fit(mnist.train.images, mnist.train.labels)
predict = clf.predict(mnist.test.images)
acc = metrics.accuracy_score(predict, mnist.test.labels)
print(acc)