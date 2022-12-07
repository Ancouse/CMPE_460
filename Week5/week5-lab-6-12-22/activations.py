import numpy as np


class Activation:
    def __call__(self, x: np.ndarray, derivative: bool = False) -> np.ndarray:
        if not derivative:
            return self._normal(x)
        else:
            return self._derivate(x)

    @staticmethod
    def _normal(x: np.ndarray):
        raise NotImplementedError

    @staticmethod
    def _derivate(x: np.ndarray):
        raise NotImplementedError

#it looks good
class Tanh(Activation):
    @staticmethod
    def _normal(x: np.ndarray):
        return np.tanh(x)

    @staticmethod
    def _derivate(x: np.ndarray):
        return 1 - np.tanh(x) ** 2

# it looks good.
class Sigmoid(Activation):
    @staticmethod
    def _normal(x: np.ndarray):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def _derivate(x: np.ndarray):
        sigmoid = 1/(1+np.exp(-x))
        return sigmoid*(1-sigmoid)

# it looks good
class ReLU(Activation):
    @staticmethod
    def _normal(x: np.ndarray):
        x[x <= 0] =0
        return x

    @staticmethod
    def _derivate(x: np.ndarray):
        x[x > 0] = 1
        x[x <= 0] = 0
        return 0

class LeakyReLU(Activation):
    @staticmethod
    def _normal(x: np.ndarray):
        x[x <= 0] = x[x <= 0] *0.1
        return x

    @staticmethod
    def _derivate(x: np.ndarray):
        x[x > 0] = 1
        x[x <= 0] = 0.1
        return x
