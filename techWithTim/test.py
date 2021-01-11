import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep=";")

print(data.head())
print()

    # We only want these columns from the dataset, G# is a grade
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

print(data.head())
print()
print("[5 rows x 7 columns]")
print()

    # We want to predicted the 3rd grade
predict = "G3"

    # Makes a new array with all the data other than what we want to predict
X = np.array(data.drop([predict], 1))       # attributes
    # Makes a new array of just the data we want tot predict
y = np.array(data[predict])        # labels
    # Split this data into 4 arrays; a test and a train set for both X and y
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

            ################## This is only needed to train new models ######################################################

#     # Keeping track of the best model to only update with better versions
# best = 0
# for _ in range(30):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

#         # Use the linear regression model to find the line of best fit on the train data
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train, y_train)
    
#     acc = linear.score(x_test, y_test)
#     print(acc)
#     if acc > best:
#         best = acc
#             # Makes a file if it doesn't already exist
#         with open("studentmodel.pickle", "wb") as f:
#                 # Saves the model as a pickle file
#             pickle.dump(linear, f)

            #################################################################################################################
    
    # Opens the file in read mode
pickle_in = open("studentmodel.pickle", "rb")
    # Loads the model into a variable called 'linear'
linear = pickle.load(pickle_in)

    # Show the gradient coefficient and y intercept
print("Coeffs: ", + linear.coef_)
print("Intercept: ", + linear.intercept_)

    # Find the accuracy of our line using the test data
acc = linear.score(x_test, y_test)
print("Accuracy: ", + acc)
print()

    # Performing predictions based on the test data
predictions = linear.predict(x_test)

    # Print a prediction for each student, the array of the rest of that students data and then their actual G3
print("----------------------------------------------------------------------------")
print("----------------------------------------------------------------------------")
for x in range(len(predictions)):
    print("Prediction:", round(predictions[x], 1), "   Actual:", y_test[x], "   Error:", round(abs(predictions[x] - y_test[x]), 1), "   Used data:", x_test[x])

    # Plotting the graph using matplotlib
p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()