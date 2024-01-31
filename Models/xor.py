import numpy as np
import pickle

from Network import losses, activations
from Network.activation_layer import ActivationLayer
from Network.fully_connected_layer import FullyConnectedLayer
from Network.network import Network

file_path = 'xor_weights'

x_input = np.asarray([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_output = np.asarray([[[0]], [[1]], [[1]], [[0]]])


def save(network: Network):
    with open(file_path, 'wb') as file:
        pickle.dump(network, file)


def load() -> Network:
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def xor_train(hidden_neurons : int = 10, epochs : int = 10000, learning_rate : float = 0.5):
    net = Network()

    net.use(losses.mse, losses.mse_prime)

    net.add(FullyConnectedLayer(2, hidden_neurons))
    net.add(ActivationLayer(activations.sigmoid, activations.sigmoid_derivative))
    net.add(FullyConnectedLayer(hidden_neurons, 1))
    net.add(ActivationLayer(activations.sigmoid, activations.sigmoid_derivative))

    net.fit(x_input, y_output, epochs, learning_rate)

    print(net.predict(x_input))

    return net


def xor_choice():
    print(err := load().error(x_input, y_output))
    best_loss = err

    for i in [10, 10, 10, 10]:
        network = xor_train(i)
        err = network.error(x_input, y_output)
        if best_loss > err:
            best_loss = err
            save(network)
            print("saved")


if __name__ == "__main__":
    net = load()
    pass