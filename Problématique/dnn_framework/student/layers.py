import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """
    w: np.ndarray
    b: np.ndarray

    def __init__(self, input_count, output_count):
        self.w = np.random.randn(output_count,input_count) * 2/(input_count+output_count)
        self.b = np.random.randn(output_count,) * 2/output_count

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
    alpha: float
    gamma: np.ndarray
    beta: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    epsilon: float = 1e-8
    def __init__(self, input_count, alpha=0.1):
        self.alpha = alpha
        self.gamma = np.zeros(input_count)
        self.beta = np.zeros(input_count)
        self.mean = np.zeros(input_count)
        self.variance = np.ones(input_count)
        Layer.train(self)
    def get_parameters(self):
        return {'gamma': self.gamma, 'beta': self.beta}


    def get_buffers(self):
        return {"global_mean":self.mean, "global_variance":self.variance}

    def forward(self, x):
        if self._is_training:
            return self._forward_training(x)
        else:
            return self._forward_evaluation(x)

    def _forward_training(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_variance = np.var(x, axis=0)
        self.mean = self.alpha * self.mean + (1 - self.alpha) * batch_mean
        self.variance = self.alpha * self.variance + (1 - self.alpha) * batch_variance
        x_norm = (x - batch_mean) / np.sqrt(batch_variance + self.epsilon)
        out = self.gamma * x_norm + self.beta
        return [out, x]

    def _forward_evaluation(self, x):
        batch_mean = self.mean
        batch_variance = self.variance
        x_norm = (x - batch_mean) / np.sqrt(batch_variance + self.epsilon)
        out = self.gamma * x_norm + self.beta

        return [out, x]

    def backward(self, output_grad, x):
        batch_mean = np.mean(x, axis=0)
        batch_variance = np.var(x, axis=0)
        x_norm = (x - batch_mean) / np.sqrt(batch_variance + self.epsilon)

        dLdx_norm = output_grad * self.gamma
        dLdvar = np.sum(dLdx_norm * (x-batch_mean) * -0.5 * np.power(batch_variance + self.epsilon, -1.5), axis=0)
        dLdmean = - np.sum(dLdx_norm / np.sqrt(batch_variance + self.epsilon), axis=0)
        dLdx = dLdx_norm / np.sqrt(batch_variance + self.epsilon) + (2 / x.shape[0]) * dLdvar * (x - batch_mean) + (dLdmean / x.shape[0])

        dLdgamma = np.sum(output_grad * x_norm,axis=0)
        dLdb = np.sum(output_grad,axis=0)

        return [dLdx,{"gamma": dLdgamma, "beta": dLdb}]




class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        result = 1 / (1 + np.exp(-x))
        return result, x

    def backward(self, output_grad, cache):
        y, _ = self.forward(cache)
        return [(y - y ** 2) * output_grad, cache]


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        return ()

    def get_buffers(self):
        # Quelques choses?
        raise NotImplementedError()

    def forward(self, x):
        y = ( x >= 0).astype(float) * x
        return y, x

    def backward(self, output_grad, cache):
        y = ( cache >= 0).astype(float) * output_grad
        return [y,cache]