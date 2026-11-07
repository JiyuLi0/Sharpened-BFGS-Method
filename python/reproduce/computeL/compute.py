"""Compute the extremal eigenvalues of H_0 saved in a MATLAB .mat file."""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.sparse import issparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_file", nargs="?", type=Path,
                        default=Path(__file__).with_name("mushrooms.mat"))
    args = parser.parse_args()
    data = loadmat(args.mat_file)
    A = data["H_0"]
    if issparse(A):
        A = A.toarray()
    eigenvalues = np.linalg.eigvalsh(A)
    print(f"μ = {eigenvalues[0]:.6f}")
    print(f"L = {eigenvalues[-1]:.6f}")


if __name__ == "__main__":
    main()
