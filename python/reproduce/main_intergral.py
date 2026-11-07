"""Run with: python -m reproduce.main_intergral --data /path/to/gisette_scale."""

import argparse
from pathlib import Path

import numpy as np

if __package__:
    from .loss_gradient import loss_gradient
    from .loss_hessian import loss_hessian
    from .normalize_row import normalize_row
else:
    from loss_gradient import loss_gradient
    from loss_hessian import loss_hessian
    from normalize_row import normalize_row


def prepare_data(X, Y):
    X = np.array(X, dtype=np.float64, copy=True)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or 0 in X.shape or X.shape[0] != len(Y):
        raise ValueError("X must be a nonempty matrix with one label per row.")
    if not np.all(np.isin(Y, [-1, 0, 1, 2])):
        raise ValueError("Supported labels: -1/1 or 0/2 (0 -> +1, 2 -> -1).")
    X[np.isnan(X)] = 0.0
    if not np.isfinite(X).all():
        raise ValueError("X contains infinite values.")
    Y = np.where((Y == 0) | (Y == 1), 1.0, -1.0)
    return normalize_row(X), Y


def newton_decrement(g, H):
    squared = float(g @ np.linalg.solve(H, g))
    if not np.isfinite(squared) or squared < 0:
        raise FloatingPointError("Invalid Newton decrement; check conditioning.")
    return np.sqrt(squared)


def bfgs_update(B, s, y):
    a = B @ s
    b, sy = float(s @ a), float(s @ y)
    if b <= 0 or sy <= 0 or not np.isfinite([b, sy]).all():
        raise FloatingPointError("Nonpositive or invalid BFGS curvature.")
    return B - np.outer(a, a) / b + np.outer(y, y) / sy


def greedy_update(B_bar, H_new):
    # np.argmax selects the first maximum, matching the MATLAB loop.
    index = int(np.argmax(np.diag(B_bar) / np.diag(H_new)))
    a, b = B_bar[:, index], H_new[:, index]
    return (B_bar - np.outer(a, a) / B_bar[index, index]
            + np.outer(b, b) / H_new[index, index])


def run_experiment(X, Y, iterations=8000, mu=1e-5, L=0.2, M=0.0,
                   tol=1e-15, verbose=True):
    """Return each method's final weights and lambda_t / lambda_0 history.

    X/Y are raw input data. Normalization and label mapping match MATLAB.
    tol=0 disables tolerance stopping; an unchanged iterate still stops.
    """
    if (iterations < 1 or not np.isfinite([mu, L, M, tol]).all()
            or mu <= 0 or L <= 0 or M < 0 or tol < 0):
        raise ValueError("Require iterations >= 1, mu/L > 0, M/tol >= 0 (finite).")
    X, Y = prepare_data(X, Y)
    d = X.shape[1]
    w_0 = np.ones(d) / (d * np.sqrt(d))
    g_0 = loss_gradient(X, Y, w_0, mu)
    H_0 = loss_hessian(X, Y, w_0, mu)
    lambda_0 = newton_decrement(g_0, H_0)
    if lambda_0 == 0:
        raise ValueError("Initial point is stationary; lambda_t/lambda_0 is undefined.")
    if verbose:
        print(f"Model Initialization Finish: N={len(Y)}, d={d}", flush=True)

    results = {}
    for method in ("gd", "bfgs", "greedy", "sharpened"):
        w, g, H = w_0.copy(), g_0.copy(), H_0.copy()
        B = L * np.eye(d)
        history = [1.0]
        status = "iteration limit"
        for iteration in range(1, iterations + 1):
            w_new = w - (g / L if method == "gd" else np.linalg.solve(B, g))
            s = w_new - w
            if not np.any(s):
                status = "unchanged iterate (floating-point precision)"
                break
            H_new = loss_hessian(X, Y, w_new, mu)
            g_new = loss_gradient(X, Y, w_new, mu)
            ratio = newton_decrement(g_new, H_new) / lambda_0
            history.append(ratio)

            # The final iterate needs no further matrix update. This also avoids
            # division by zero once an experiment reaches numerical convergence.
            if ratio <= tol or iteration == iterations:
                w = w_new
                if ratio <= tol:
                    status = "tolerance reached"
                break

            if method in ("bfgs", "sharpened"):
                y = (0.5 * (H + H_new)) @ s
                B = bfgs_update(B, s, y)
            if method in ("greedy", "sharpened"):
                r = np.sqrt(s @ H @ s)
                factor = 1 + M * r if method == "greedy" else (1 + 0.5 * M * r)**2
                B = greedy_update(factor * B, H_new)
            w, g, H = w_new, g_new, H_new
            if verbose and iteration % 100 == 0:
                print(f"{method}: {iteration}/{iterations}, ratio={ratio:.3e}", flush=True)
        results[method] = {"w": w, "history": np.asarray(history), "status": status}
        if verbose:
            print(f"{method}: {len(history)-1} steps, ratio={history[-1]:.3e}; {status}",
                  flush=True)
    return results


def save_results(results, output, iterations, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {}
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, label, color in (("bfgs", "BFGS", "#0000FF"),
                                 ("gd", "GD", "#000000"),
                                 ("greedy", "Greedy", "#FFA500"),
                                 ("sharpened", "Sharpened", "#FF5733")):
        result = results[method]
        history = result["history"]
        ax.semilogy(np.arange(len(history)), np.maximum(history, np.finfo(float).tiny),
                    "-.", color=color, linewidth=2, label=label)
        arrays[f"y_{method}"] = history
        arrays[f"w_{method}"] = result["w"]
        arrays[f"status_{method}"] = np.asarray(result["status"])
    ax.legend(loc="lower left")
    ax.set_xlabel(r"Number of iterations $t$")
    ax.set_ylabel(r"$\lambda_f(x_t)/\lambda_f(x_0)$")
    ax.set_xlim(0, iterations)
    ax.set_ylim(1e-15, 1)
    fig.tight_layout()
    fig.savefig(output / "convergence.png", dpi=200)
    np.savez(output / "results.npz", **arrays)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved results to {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", type=Path, default=Path("gisette_scale"),
                        help="LIBSVM-format input file (default: ./gisette_scale)")
    source.add_argument("--demo", action="store_true", help="Use small synthetic data")
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--mu", type=float, default=1e-5)
    parser.add_argument("--L", type=float, default=0.2)
    parser.add_argument("--M", type=float, default=0.0)
    parser.add_argument("--tol", type=float, default=1e-15)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--show", action="store_true", help="Also display the plot")
    args = parser.parse_args()
    if args.demo:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 10))
        Y = np.where(X @ rng.normal(size=10) + rng.normal(size=200) >= 0, 1, -1)
    else:
        if not args.data.is_file():
            parser.error(f"Data file not found: {args.data}. Supply --data PATH or use --demo.")
        from sklearn.datasets import load_svmlight_file
        X_sparse, Y = load_svmlight_file(str(args.data), zero_based=False)
        X = X_sparse.toarray()
    results = run_experiment(X, Y, iterations=args.iterations, mu=args.mu,
                             L=args.L, M=args.M, tol=args.tol)
    save_results(results, args.output, args.iterations, show=args.show)


if __name__ == "__main__":
    main()
