import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

    # Preprocessing the data due to high amount of non numerical data
le = preprocessing.LabelEncoder()
    # Creating a list for each column
    # All values are converted to numerical (e.g. low > 0, med > 1, high > 2)
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))    # features
y = list(cls)    # labels

    # Split the data into the 4 groups
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

        ####################### Edit the number of neighbors #####################################################
    # Creating a classifier
model = KNeighborsClassifier(n_neighbors=9)

    # Training the model
model.fit(x_train, y_train)
acc = round(model.score(x_test, y_test), 3) * 100

    # Make some predictions on the test data
predicted = model.predict(x_test)
    # Giving names for the classifier so we see the names and not just numerical values as transformed above
names = ["acc", "good", "unacc", "vgood"]

for x in range(len(x_test)):
    print("Predicted: ", names[predicted[x]], "   Actual: ", names[y_test[x]], "   Used data: ", x_test[x])
    n = model.kneighbors([x_test[x]], 3, True)

print("Accuracy:", acc, "%")