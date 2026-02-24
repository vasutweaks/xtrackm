import argparse

import matplotlib.pyplot as plt
import pandas as pd


default="std_oscar_track_u_coastal.csv"
df = pd.read_csv(default)
x = df.index
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, df["std_oscar"], label="std_oscar", color="tab:blue", linewidth=1.8)
ax.plot(x, df["std_track"], label="std_track", color="tab:red", linewidth=1.8)
ax.set_xlabel("Index")
ax.set_ylabel("Std (m/s)")
ax.set_title("std_oscar and std_track vs index")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
plt.savefig("std_oscar_track_u_coastal.png")
plt.show()
# fig.savefig(args.out, dpi=180)
