from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from classifier import KNNClassifier
from sklearn.metrics import accuracy_score


def load_data():
    dataset = load_iris()
    X, y = pd.DataFrame(dataset.data), pd.Series(dataset.target)
    X.columns = dataset.feature_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
    return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = load_data()

    model = KNNClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"KNN Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
