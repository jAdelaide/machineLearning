import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

    # Gamma = learning rate
clf = svm.SVC(gamma=0.001, C=100)

x,y = digits.data[:-10], digits.target[:-10]
clf.fit(x,y)

print('Prediction:',clf.predict(digits.data[[-57]]))

plt.imshow(digits.images[-57], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()

