from abc import ABC, abstractmethod


class AbstractClassifier(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, features, labels):
        pass

    @abstractmethod
    def predict(self, features):
        pass
