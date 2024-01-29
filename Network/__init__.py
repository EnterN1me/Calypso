import numpy as np

import activations
import losses
from activation_layer import ActivationLayer
from fully_connected_layer import FullyConnectedLayer
from network import Network


def xor_test():
    net = Network()

    net.use(losses.mse, losses.mse_prime)

    net.add(FullyConnectedLayer(2, 2))
    net.add(ActivationLayer(activations.sigmoid, activations.sigmoid_derivative))
    net.add(FullyConnectedLayer(2, 1))
    net.add(ActivationLayer(activations.sigmoid, activations.sigmoid_derivative))

    x_input = np.asarray([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_output = np.asarray([[[0]], [[1]], [[1]], [[0]]])

    net.fit(x_input, y_output, 100, 0.1)

    print(net.predict(x_input))


if __name__ == "__main__":
    xor_test()
    pass
