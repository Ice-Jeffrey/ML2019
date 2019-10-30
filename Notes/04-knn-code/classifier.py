import pandas as pd
import numpy as np
from abstract_classifier import AbstractClassifier
from learning_lib import distance_metric


class KNNClassifier(AbstractClassifier):
    def fit(self, X_train, y_train):
        # There is no "training" step in kNN
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for _, instance in X_test.iterrows():
            predictions.append(self.k_nearest_neighbor(instance, k=3))
        return np.array(predictions)

    def k_nearest_neighbor(self, target, k=1):
        # Build an indexed distance map
        indexes = {}
        for i, instance in self.X_train.iterrows():
            indexes[i] = distance_metric(instance, target)
        # Sort the map by distance (ascending) and take the shortest k instances
        k_nearest_dists = sorted(indexes.items(), key=lambda kv: kv[1])[:k]
        # Throw away the distances leaving the indexes
        k_nearest_indexes = list(map(lambda d: d[0], k_nearest_dists))
        # Collect the training classes associated with the predicted indexes
        counts = np.bincount(np.array(self.y_train[k_nearest_indexes]))
        # Return the most frequently occurring class as the winning predicted class
        return np.argmax(counts)
