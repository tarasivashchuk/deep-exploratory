from tensorflow.python.layers.base import Layer


# This code was taken from Marco Cerliani. I copy and pasted it from his blog post:
# https://towardsdatascience.com/time2vec-for-time-series-features-encoding-a03a4f3f937e


class T2V(Layer):

    def __init__(self, output_dim=None, input_dim=None, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        super(T2V, self).__init__(**kwargs)

    def build(self, input_shape):        self.W = self.add_weight(name='W',
                                                                  shape=(
                                                                  self.output_dim,
                                                                  self.output_dim),
                                                                  initializer='uniform',
                                                                  trainable=True)

    self.B = self.add_weight(name='B',
                             shape=(self.input_dim, self.output_dim),
                             initializer='uniform', trainable=True)
    self.w = self.add_weight(name='w',
                             shape=(1, 1),
                             initializer='uniform', trainable=True)
    self.b = self.add_weight(name='b',
                             shape=(self.input_dim, 1),
                             initializer='uniform', trainable=True)
    super(T2V, self).build(input_shape)

    def call(self, x):
        original = self.w * x + self.b
        x = K.repeat_elements(x, self.output_dim, -1)
        sin_trans = K.sin(K.dot(x, self.W) + self.B)
        return K.concatenate([sin_trans, original], -1)

        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[1], self.output_dim + 1)