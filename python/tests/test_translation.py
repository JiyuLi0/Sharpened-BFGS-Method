import unittest

import numpy as np
from numpy.testing import assert_allclose

from reproduce.loss_function import loss_function
from reproduce.loss_gradient import loss_gradient
from reproduce.loss_hessian import loss_hessian
from reproduce.main_intergral import prepare_data, run_experiment


def matlab_reference(X, Y, method, iterations, mu, L, M):
    """Literal small-input reference using the original exponential formulas."""
    N, d = X.shape

    def gradient(w):
        h = np.exp(-Y * (X @ w))
        return np.sum((-Y * h / (1 + h))[:, None] * X, axis=0) / N + mu * w

    def hessian(w):
        h = np.exp(-Y * (X @ w))
        return X.T @ ((h * Y * Y / ((1 + h)**2))[:, None] * X) / N + mu * np.eye(d)

    w = np.ones(d) / (d * np.sqrt(d))
    B = L * np.eye(d)
    g = gradient(w)
    lambda_0 = np.sqrt(g @ np.linalg.solve(hessian(w), g))
    history = [1.0]
    for _ in range(iterations):
        w_new = w - (gradient(w) / L if method == "gd"
                     else np.linalg.solve(B, gradient(w)))
        s = w_new - w
        H, H_new = hessian(w), hessian(w_new)
        if method in ("bfgs", "sharpened"):
            y = (0.5 * (H + H_new)) @ s
            a = B @ s
            B = B - np.outer(a, a) / (s @ a) + np.outer(y, y) / (s @ y)
        if method in ("greedy", "sharpened"):
            r = np.sqrt(s @ H @ s)
            B_bar = ((1 + M * r) if method == "greedy" else (1 + 0.5 * M * r)**2) * B
            index = 0
            max_res = B_bar[0, 0] / H_new[0, 0]
            for i in range(1, d):
                temp = B_bar[i, i] / H_new[i, i]
                if temp > max_res:
                    max_res, index = temp, i
            a, b = B_bar[:, index], H_new[:, index]
            B = (B_bar - np.outer(a, a) / B_bar[index, index]
                 + np.outer(b, b) / H_new[index, index])
        w = w_new
        g = gradient(w)
        history.append(np.sqrt(g @ np.linalg.solve(hessian(w), g)) / lambda_0)
    return w, history


class TranslationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.raw_X = rng.normal(size=(30, 4))
        self.Y = rng.choice([-1.0, 1.0], size=30)
        self.X, _ = prepare_data(self.raw_X, self.Y)

    def test_derivatives(self):
        w = np.array([0.2, -0.1, 0.4, 0.3])
        mu, eps = 0.03, 1e-5
        directions = np.eye(4) * eps
        gradient_fd = np.array([
            (loss_function(self.X, self.Y, w + e, mu)
             - loss_function(self.X, self.Y, w - e, mu)) / (2 * eps)
            for e in directions])
        hessian_fd = np.column_stack([
            (loss_gradient(self.X, self.Y, w + e, mu)
             - loss_gradient(self.X, self.Y, w - e, mu)) / (2 * eps)
            for e in directions])
        assert_allclose(loss_gradient(self.X, self.Y, w, mu), gradient_fd, atol=1e-10)
        assert_allclose(loss_hessian(self.X, self.Y, w, mu), hessian_fd, atol=1e-10)

    def test_matlab_trajectories(self):
        for M in (0.0, 0.3):
            results = run_experiment(self.raw_X, self.Y, iterations=5, mu=0.03,
                                     L=0.2, M=M, tol=0, verbose=False)
            for method, result in results.items():
                with self.subTest(method=method, M=M):
                    w, history = matlab_reference(self.X, self.Y, method, 5, 0.03, 0.2, M)
                    assert_allclose(result["w"], w, rtol=1e-10, atol=1e-12)
                    assert_allclose(result["history"], history, rtol=1e-9, atol=1e-12)

    def test_preprocessing(self):
        X = np.array([[3, 4], [0, 0], [np.nan, 2], [1, 0]])
        normalized, Y = prepare_data(X, [0, 2, 1, -1])
        assert_allclose(normalized, [[0.6, 0.8], [0, 0], [0, 1], [1, 0]])
        assert_allclose(Y, [1, -1, 1, -1])
        self.assertTrue(np.isnan(X[2, 0]))  # Caller input is not changed.

    def test_large_margins_are_finite(self):
        X, Y, w = np.eye(2), np.ones(2), np.array([1000., -1000.])
        with np.errstate(over="raise", invalid="raise"):
            self.assertTrue(np.isfinite(loss_function(X, Y, w, 0.01)))
            self.assertTrue(np.isfinite(loss_gradient(X, Y, w, 0.01)).all())
            self.assertTrue(np.isfinite(loss_hessian(X, Y, w, 0.01)).all())

    def test_convergence_stops_before_invalid_update(self):
        result = run_experiment(self.raw_X, self.Y, iterations=100, mu=0.03,
                                tol=1e-12, verbose=False)
        for method in ("bfgs", "greedy", "sharpened"):
            self.assertEqual(result[method]["status"], "tolerance reached")
            self.assertLessEqual(result[method]["history"][-1], 1e-12)


if __name__ == "__main__":
    unittest.main()
