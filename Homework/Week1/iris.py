import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from learning_lib import accuracy_score

from bow_classifier import IrisClassifier


def load_data():
    iris = datasets.load_iris()
    data = np.hstack((iris.data, iris.target.reshape(-1, 1)))
    data = pd.DataFrame(data)
    data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    return data


def main():
    # 1. Load and transform
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
   
    # 2. Train
    model = IrisClassifier()
    model.fit(X_train, y_train)

    # All test data
    predictions = {}
    for (i, message) in enumerate(X_test):
        predictions[i] = model.predict(message)
    print(accuracy_score(y_test, predictions))
    

if __name__ == "__main__":
    main()
