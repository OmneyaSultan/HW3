from abc import ABCMeta, abstractmethod
import pickle


class RetrievalModel(object, metaclass=ABCMeta):
    def __init__(self, model_file):
        self.model_file = model_file

    def save_model(self):
        with open(self.model_file, "wb") as file:
            pickle.dump(self, file)

    def load_model(self):
        with open(self.model_file, "rb") as file:
            model = pickle.load(file)
        return model

    @abstractmethod
    def index(self, input_file):
        pass

    @abstractmethod
    def search(self, query, k):
        pass
