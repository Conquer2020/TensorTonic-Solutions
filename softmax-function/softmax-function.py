import numpy as np


def softmax(x: list) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    maximum = np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(values - maximum)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)
