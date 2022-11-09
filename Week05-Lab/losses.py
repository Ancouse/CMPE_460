import numpy as np


class Loss:
    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray, derivative: bool = False
    ) -> float:
        if not derivative:
            return self._normal(y_true, y_pred)
        else:
            return self._derivate(y_true, y_pred)

    @staticmethod
    def _normal(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError

    @staticmethod
    def _derivate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError


class MSE(Loss):
    @staticmethod
    def _normal(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.power(y_true - y_pred, 2))

    @staticmethod
    def _derivate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return 2 * (y_pred - y_true) / y_true.size
