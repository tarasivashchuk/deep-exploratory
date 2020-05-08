import numpy as np


class Dense(object):
    def __init__(self, input_shape, layer_size, activation):
        self.weights = np.random.randn(layer_size, input_shape)
        self.bias = np.zeros((layer_size, 1))
        self.activation = activation

    def forward(self, input_data):
        # TODO: Switch around the shapes so that weights are transposed
        input_data = np.reshape(input_data, (1, 20))
        return self.activation(np.add(np.dot(self.weights, input_data.T), self.bias))