import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)

    # Implement the support vector classifier
    ######################### Select parameters here #####################################################
    # C = margin (how many points between the nearest measured values)
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = round(metrics.accuracy_score(y_test, y_pred), 3) * 100
print("SVC accuracy:", acc, "%")

clf = KNeighborsClassifier(n_neighbors=12)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = round(metrics.accuracy_score(y_test, y_pred), 3) * 100
print("KNN accuracy:", acc, "%")