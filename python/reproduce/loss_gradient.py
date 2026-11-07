from scipy.special import expit


def loss_gradient(X, Y, w, mu):
    h = expit(-Y * (X @ w))
    return X.T @ (-Y * h) / len(Y) + mu * w
