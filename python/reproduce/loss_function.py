import numpy as np


def loss_function(X, Y, w, mu):
    """Mean logistic loss with L2 regularization; vectors have shape (n,)."""
    return float(np.mean(np.logaddexp(0.0, -Y * (X @ w))) + mu * (w @ w) / 2)
