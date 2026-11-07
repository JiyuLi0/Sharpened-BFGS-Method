import numpy as np


def normalize_row(A):
    A = np.asarray(A, dtype=np.float64)
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.where(norms == 0, 1.0, norms)
