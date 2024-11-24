from abc import ABC, abstractmethod


class SingleBOTraining(ABC):

    @abstractmethod
    def generate_initial_data(self, n=10):
        pass

    @abstractmethod
    def initialize_model(self, train_x, train_obj):
        pass

    @abstractmethod
    def optimize_acquisition_and_get_observation(self, acq_func):
        pass

    @abstractmethod
    def train(self, verbose=False):
        pass
