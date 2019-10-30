from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Training and test matrices, training and test vectors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Classifiers
model = neighbors.KNeighborsClassifier()


model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"KNN Accuracy: {accuracy}")

