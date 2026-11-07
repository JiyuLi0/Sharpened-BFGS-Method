import numpy as np
from scipy.special import expit


def loss_hessian(X, Y, w, mu):
    margin = Y * (X @ w)
    weights = Y**2 * expit(margin) * expit(-margin)
    return X.T @ (weights[:, None] * X) / len(Y) + mu * np.eye(X.shape[1])
