import numpy as np

from util import generate_data


class Dense(object):
    def __init__(self, input_shape, layer_size, activation):
        self.weights = np.random.randn(layer_size, input_shape)
        self.bias = np.zeros((layer_size, 1))
        self.activation = activation

    def forward(self, input_data):
        # TODO: Switch around the shapes so that weights are transposed
        input_data = np.reshape(input_data, (1, 20))
        return self.activation(np.add(np.dot(self.weights, input_data.T), self.bias))


if __name__ == '__main__':
    x, y = generate_data.normal_dataset(0., 10., 20)
    tanh = lambda output: np.tanh(output)
    linear = lambda output: output
    # print(x.shape)
    layer_1 = Dense(x.shape[0], 5, linear)
    print(layer_1.forward(x))

# data_weight = 0.2
#
#
# def neuron(nn_input, nn_weight):
#     return nn_input * nn_weight
#
#
# for example in data:
#     print(neuron(example, data_weight))
