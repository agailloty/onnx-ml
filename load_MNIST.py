from sklearn import datasets
import numpy as np

mnist = datasets.load_digits()

np.savetxt("mnist_train.csv", mnist.data, delimiter = ",")
np.savetxt("mnist_label.csv", mnist.target, delimiter = ",")
