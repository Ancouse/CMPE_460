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
        ############# YOUR CODE #############
        pass

    @staticmethod
    def _derivate(x: np.ndarray):
        ############# YOUR CODE #############
        pass


class ReLU(Activation):
    @staticmethod
    def _normal(x: np.ndarray):
        ############# YOUR CODE #############
        pass

    @staticmethod
    def _derivate(x: np.ndarray):
        ############# YOUR CODE #############
        pass


class LeakyReLU(Activation):
    @staticmethod
    def _normal(x: np.ndarray):
        ############# YOUR CODE #############
        pass

    @staticmethod
    def _derivate(x: np.ndarray):
        ############# YOUR CODE #############
        pass
