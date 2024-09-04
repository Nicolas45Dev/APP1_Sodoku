import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """
    w: np.ndarray
    b: np.ndarray
    input: np.ndarray
    output: np.ndarray

    def __init__(self, input_count, output_count):
        self.input = np.zeros(input_count)
        self.output = np.zeros(output_count)
        self.w = np.zeros(input_count)
        self.b = np.zeros(1)

    def get_parameters(self):
        return {'w': self.w, 'b': self.b}

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        return [x @ self.w.T + self.b, x]

    def backward(self, output_grad, cache):
        x_grad = output_grad @ self.w
        b_grad = np.sum(output_grad, axis=0)
        w_grad = output_grad.T @ cache

        return [x_grad,{"w": w_grad, "b": b_grad}]


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        raise NotImplementedError()

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def _forward_training(self, x):
        raise NotImplementedError()

    def _forward_evaluation(self, x):
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        raise NotImplementedError()


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        raise NotImplementedError()


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        raise NotImplementedError()
