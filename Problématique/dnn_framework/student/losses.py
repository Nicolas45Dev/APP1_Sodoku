import numpy as np

from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    """
    This class combines a softmax activation function and a cross entropy loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C))
        :param target: The target classes (shape: (N,))
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        # The error is -Sum(p(x) * log(q(x))), where p(x) is the target distribution and q(x) is the predicted distribution
        probabilities = softmax(x)
        n_echantillons = x.shape[0]

        p = -np.log(probabilities[range(n_echantillons), target])
        loss = np.sum(p) / n_echantillons

        input_grad = probabilities.copy()
        input_grad[range(n_echantillons), target] -= 1
        input_grad /= n_echantillons

        return loss, input_grad


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class MeanSquaredErrorLoss(Loss):
    """
    This class implements a mean squared error loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: any)
        :param target: The target tensor (shape: same as x)
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        error = np.mean((x - target) ** 2)
        return error, 2 * (x - target) / x.size
