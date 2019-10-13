import pandas as pd
import numpy as np
from sklearn import datasets, model_selection, naive_bayes as nb

from learning_lib import train_test_split, accuracy_score
from nlp import transform
from bow_classifier import BagOfWordsClassifier


def load_data():
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target.reshape(-1, 1)
    B = []
    for i in range(data.shape[0]):
        temp = ''
        for j in range(data.shape[1]):
            temp += str(data[i][j]) + ' '
        B.append(temp)
    B = np.array(B).reshape(-1, 1)
    data = pd.DataFrame(np.hstack((target, B)))
    data.columns = ['class', 'message']
    return data


def main_1():
    # 1. Load and transform
    data = load_data()
    data['message'] = data['message'].apply(transform)

    # 2. Visualize
    train_data, test_data = train_test_split(data, test_size=0.2)
    
    # 3. Train
    c = BagOfWordsClassifier()
    c.fit(train_data['message'], train_data['class'])

    # 4. Test all test data
    predictions = {}
    for (i, message) in enumerate(test_data['message']):
        predictions[i] = c.predict(message)
    print(accuracy_score(test_data['class'], predictions))
    

def main_2():
    from sklearn.metrics import accuracy_score as ase
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data, target, test_size=0.2, random_state=42)
    model = nb.MultinomialNB()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print(ase(y_test, prediction))

if __name__ == "__main__":
    main_1()
    print("\n")
    main_2()
