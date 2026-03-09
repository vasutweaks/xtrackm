#!/usr/bin/env python3
"""Small demo of row-wise broadcasting behavior in pandas column assignment."""

import pandas as pd


# Base table: row-varying columns
df = pd.DataFrame(
    {
        "time": pd.to_datetime(
            ["2026-01-01 00:00", "2026-01-01 06:00", "2026-01-01 12:00"]
        ),
        "gu": [0.15, 0.31, 0.22],
        "rama_u": [0.12, 0.29, 0.20],
    }
)

print("Original DataFrame:")
print(df, end="\n\n")

# Scalar assignment: pandas broadcasts one value to all rows.
sat = "S3A"
matched_pairs = len(df)
df["sat"] = sat
df["matched_pairs"] = matched_pairs

print("After scalar assignment (broadcast to every row):")
print(df[["time", "gu", "rama_u", "sat", "matched_pairs"]], end="\n\n")

# Sequence assignment: one value per row (no broadcast).
df["quality_flag"] = ["good", "ok", "good"]
print("After sequence assignment (row-by-row values):")
print(df[["time", "quality_flag"]], end="\n\n")

# Wrong-length sequence assignment raises an error.
try:
    df["bad_col"] = [1, 2]  # len=2 for 3-row DataFrame
except ValueError as exc:
    print("Wrong-length assignment error:")
    print(f"{type(exc).__name__}: {exc}")
