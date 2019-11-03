from layers import *

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
