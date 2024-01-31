from Network.layer import Layer
import numpy as np


class FullyConnectedLayer(Layer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.normal(0, 0.5, size=(input_size, output_size))
        self.bias = np.random.normal(0, 0.5, size=(1, output_size))

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

    def __str__(self):
        return "Fully Connected Layer {} inputs and {} neurons".format(self.input_size,self.output)
