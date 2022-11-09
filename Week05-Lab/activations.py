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


class Tanh(Activation):
    @staticmethod
    def _normal(x: np.ndarray):
        return np.tanh(x)

    @staticmethod
    def _derivate(x: np.ndarray):
        return 1 - np.tanh(x) ** 2


class Sigmoid(Activation):
    @staticmethod
    def _normal(x: np.ndarray):
        s = 1/(1 + np.exp(-x))
        return s

    @staticmethod
    def _derivate(x: np.ndarray):
        s = 1/(1 + np.exp(-x))
        ds = s*(1-s)
        return ds


class ReLU(Activation):
    @staticmethod
    def _normal(x: np.ndarray):
        return np.maximum(0., x)

    @staticmethod
    def _derivate(x: np.ndarray):
        return np.greater(x, 0).astype(np.float32)


class LeakyReLU(Activation):
    @staticmethod
    def _normal(x: np.ndarray):
        return x*0.01 if x < 0 else x

    @staticmethod
    def _derivate(x: np.ndarray):
        return 0.01 if x < 0 else 1
