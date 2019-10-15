from abc import ABC, abstractmethod

class AbstractIrisClassifier(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, messages, labels):
        pass

    @abstractmethod
    def predict(self, message):
        pass