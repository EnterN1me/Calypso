from typing import Callable, Any
from Network.layer import Layer
import numpy as np

class Network:

    layers = list[Any]

    loss_function = Callable[[np.ndarray,np.ndarray],np.ndarray]

    loss = loss_function | None
    loss_prime = loss_function | None

    def add(self, layer: Layer) -> None: ...
    def use(self, loss: loss_function, loss_prime: loss_function) -> None: ...
    def predict(self, input_data: np.ndarray) -> list[Any]: ...
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, learning_rate: float = ...) -> None: ...