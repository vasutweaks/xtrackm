#!/usr/bin/env python3
"""Plot ODO filtering-window shapes for multiple (p, q) combinations."""

import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt


def get_optimal_weights(p: int, q: int, dt: float = 1.0):
    """Derive Powell & Leben ODO coefficients c_n."""
    p_abs, q_abs = abs(p), abs(q)
    N = p_abs + q_abs
    T = N + 1
    n = np.array([i for i in range(-p_abs, q_abs + 1) if i != 0], dtype=float)

    A = np.zeros((T, T), dtype=float)
    A[N, :N] = 1.0
    for i in range(N):
        A[i, :N] = (1.0 / n) * (-n[i] / 2.0)
        A[i, N] = (n[i] ** 2 * dt ** 2) / 4.0
        A[i, i] = -1.0

    B = np.zeros(T, dtype=float)
    B[N] = 1.0
    results = np.linalg.solve(A, B)
    return results[:N], n


def parse_int_list(values: str):
    return [int(v.strip()) for v in values.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Plot ODO filtering-window shape for different (p, q) combinations.")
    parser.add_argument("--p", default="3,5", help="Comma-separated p values (example: 3,5,7)")
    parser.add_argument("--q", default="3,5", help="Comma-separated q values (example: 3,5,7)")
    parser.add_argument("--dt", type=float, default=1.0, help="Sampling interval dt")
    parser.add_argument("--output", default="odo_filter_windows.png", help="Output figure path")
    parser.add_argument("--show", action="store_true", help="Show figure interactively")
    args = parser.parse_args()

    p_values = parse_int_list(args.p)
    q_values = parse_int_list(args.q)

    if not p_values or not q_values:
        raise ValueError("Both --p and --q must include at least one integer.")

    combos = list(itertools.product(p_values, q_values))

    fig, ax = plt.subplots(figsize=(10, 6))
    for p, q in combos:
        cn, n = get_optimal_weights(-abs(p), abs(q), args.dt)
        ax.plot(n, cn, marker="o", linewidth=1.5, label=f"p={-abs(p)}, q={abs(q)}")

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Relative index n")
    ax.set_ylabel("Weight c_n")
    ax.set_title(f"ODO filtering-window shape (dt={args.dt})")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)

    if args.show:
        plt.show()

    print(f"Saved plot: {args.output}")
    print("Combinations:", ", ".join([f"(p={-abs(p)},q={abs(q)})" for p, q in combos]))


if __name__ == "__main__":
    main()
