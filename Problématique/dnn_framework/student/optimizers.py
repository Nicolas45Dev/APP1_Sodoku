from dnn_framework.optimizer import Optimizer


class SgdOptimizer(Optimizer):
    """
    This class implements a stochastic gradient descent optimizer.
    """
    learning_rate: float
    _parameters: dict

    def __init__(self, parameters, learning_rate=0.01):
        self._parameters = parameters
        self.learning_rate = learning_rate

    def _step_parameter(self, parameter, parameter_grad, parameter_name):
        return parameter - self.learning_rate * parameter_grad


class AdamOptimizer(Optimizer):

    learning_rate: float
    _parameters: dict
    beta1: float
    beta2: float
    epsilon: float
    _m: dict
    _v: dict
    _t: int

    def __init__(self, parameters, learning_rate=0.001, beta1=0.89, beta2=0.95, epsilon=1e-8):
        self._parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.t = 0
        # Initialze the m moment and the v moment to 0
        self._initialize_moment()

    # Initialize the m moment and the v moment to 0
    def _initialize_moment(self):
        self._m = {name: 0 for name in self._parameters}
        self._v = {name: 0 for name in self._parameters}

    def _step_parameter(self, parameter, parameter_grad, parameter_name):

        # Clear the moments
        self._initialize_moment()

        self.t += 1

        # Update the first moment
        self._m[parameter_name] = self. beta1 * self._m[parameter_name] + (1 - self.beta1) * parameter_grad

        # Update the second moment
        self._v[parameter_name] = self.beta2 * self._v[parameter_name] + (1 - self.beta2) * parameter_grad ** 2

        # Bias correction (m)
        m_bias_corrected = self._m[parameter_name] / (1 - self.beta1 ** self.t)

        # Bias correction (v)
        v_bias_corrected = self._v[parameter_name] / (1 - self.beta2 ** self.t)

        parameter -= self.learning_rate * m_bias_corrected / (v_bias_corrected ** 0.5 + self.epsilon)

        return parameter
