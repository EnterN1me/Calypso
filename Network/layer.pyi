from abc import ABC

import numpy as np

class Layer(ABC):

    input: np.ndarray | None
    output: np.ndarray | None

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray: ...
    def backward_propagation(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray: ...
