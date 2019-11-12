from sklearn.metrics import *
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

# Load dataset
iris = datasets.load_diabetes()
X = iris.data
y = iris.target

# Training and test matrices, training and test vectors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Regressor
tree = neighbors.KDTree(X_train)
trainresult = tree.query(X_test,5,return_distance=False)
predictions1 = []
for re in trainresult:
    #print(re)
    counts = np.average(np.array(y_train[re]))
    predictions1.append(counts)

model = neighbors.KNeighborsRegressor()
model.fit(X_train, y_train)
predictions2 = model.predict(X_test)

# print the accuracy score
accuracy1 = mean_squared_error(predictions1, y_test)
print(accuracy1)
accuracy2 = mean_squared_error(predictions2, y_test)
print(accuracy2)