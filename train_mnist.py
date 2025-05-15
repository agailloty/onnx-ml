import numpy as np

mnist_data = np.loadtxt("mnist_train.csv", delimiter=",")
mnist_labels = np.loadtxt("mnist_label.csv", delimiter=",")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mnist_data, mnist_labels, test_size=0.10)

from sklearn import svm
svm_classifier = svm.SVC(gamma=0.001) 
#fit to the trainin data
svm_classifier.fit(X_train,y_train)
y_pred = svm_classifier.predict(X_test)

from sklearn.metrics import accuracy_score

print("Accuracy of model = %2f%%" % (accuracy_score(y_test, y_pred )*100))

# Convert into ONNX format.
from skl2onnx import to_onnx

onx = to_onnx(svm_classifier, mnist_data[1:])
with open("mnist_svm.onnx", "wb") as f:
    f.write(onx.SerializeToString())