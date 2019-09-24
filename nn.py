"""
This is a file to test initial code.
"""

data = [5.5, 10.6, 15.4]
data_weight = 0.2


def neuron(nn_input, nn_weight):
    return nn_input * nn_weight


for example in data:
    print(neuron(example, data_weight))
