import numpy as np
import sys

p = -5
q = 6
dt: float = 1.0
p_abs, q_abs = abs(p), abs(q)
N = p_abs + q_abs
T = N + 1
# indices from -p to q excluding 0
# Relative indices excluding the current point (0)
n = np.array([i for i in range(-p_abs, q_abs + 1) if i != 0], dtype=float)
# Build Lagrange Matrix A [cite: 471-478]
A = np.zeros((T, T), dtype=float)
# to make bottom rows one; except the last element of bottom row
A[N, :N] = 1.0
# print(A)
for i in range(N):
    # c_n derivation from the Lagrange function [cite: 85, 477]
    # assinging all elements of each row (except last element)
    # the RHS is vector of size N where A's size is N+1 by N+1
    A[i, :N] = (1.0 / n) * (-n[i] / 2.0)
    # print(i, (1.0 / n)*(-n[i] / 2.0))
    # assinging last element of each row
    A[i, N] = (n[i] ** 2 * dt ** 2) / 4.0
    print((n[i] ** 2 * dt ** 2) / 4.0)
    A[i, i] = -1.0

B = np.zeros(T, dtype=float)
B[N] = 1.0
# print(B)
results = np.linalg.solve(A, B)
weights = results[:N]
print(weights)
sys.exit()
