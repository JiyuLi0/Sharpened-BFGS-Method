from scipy.io import loadmat
data = loadmat('./reproduce/computeL/mushrooms.mat')
import numpy as np
print(data.keys())
A=data['H_0']
eigenvalues = np.linalg.eigvalsh(A)  
mu = np.min(eigenvalues)
L = np.max(eigenvalues)
print(f"μ = {mu:.6f}")
print(f"L = {L:.6f}")
