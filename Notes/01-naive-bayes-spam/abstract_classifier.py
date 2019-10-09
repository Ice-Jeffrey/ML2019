from abc import ABC, abstractmethod

class AbstractSPAMClassifier(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, messages, labels):
        pass

    @abstractmethod
    def predict(self, message):
        pass

