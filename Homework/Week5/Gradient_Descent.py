from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_data():
    dataset = datasets.load_boston()
    data = pd.DataFrame(dataset.data)
    target = pd.DataFrame(dataset.target)
    data.cols = dataset.feature_names
    X_train, X_test, y_train, y_test = train_test_split(data, target, shuffle=True)
    return X_train, X_test, y_train, y_test

def LossFunction(x, y, theta):
    y_predict = theta[0] + theta[1] * x
    return (y_predict - y) ** 2

def GradientDescent(X, y, theta, learning_rate, num_iter):
    X = np.array(X)
    y = np.array(y)
    costs = []
    for i in range(num_iter):
        gradient = [0, 0]
        cost = 0
        for j in range(len(X)):
            y_predict = theta[0] + theta[1] * X[j]
            gradient[0] += (y_predict - y[j])
            gradient[1] += (y_predict - y[j]) * X[j]
            cost += LossFunction(X[j], y[j], theta)
        cost = cost / (2 * len(X))
        theta[0] = theta[0] - learning_rate * gradient[0]
        theta[1] = theta[1] - learning_rate * gradient[1]
        costs.append(cost)
        print(cost)
    return theta, costs

def FindBestFeature(data, target):
    for feature in data.columns:
        plt.scatter(data[feature], target)
        plt.show()
    best_feature = data.columns[5]
    plt.scatter(data[best_feature], target)
    plt.show()
    return best_feature

def main():
    # 1. load the datasets
    X_train, X_test, y_train, y_test = load_data()

    # 2. decide whichi feature to use for gradient descent
    bestFeature = FindBestFeature(X_train, y_train)

    # 3. define the learning rate and the initial parameters for gradient descent
    learning_rate = 0.0001
    theta = np.zeros(2)
    num_iter = 20000
    theta, costs = GradientDescent(X_train[bestFeature], y_train, theta, learning_rate, num_iter)

    # 4. plot the final graph
    rx = np.array(X_test.loc[:, bestFeature])
    ry = np.array(y_test)
    plt.grid()
    plt.scatter(rx, ry)
    y = theta[0] + rx * theta[1]
    plt.plot(rx, y, '-r', label=f'y={theta[1]}x+{theta[0]}')
    plt.show()


if __name__ == "__main__":
    main()