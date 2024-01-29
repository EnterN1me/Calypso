from abc import ABC, abstractmethod, ABCMeta


class Layer(ABC):

    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward_propagation(self, input_data):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    @abstractmethod
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
