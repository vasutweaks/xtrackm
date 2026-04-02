import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Input file produced by your processing script
nc_file = "counts_all_S3A_xtrack.nc"

ds = xr.open_dataset(nc_file)
counts_all = ds["counts_all"]

# Mean across tracks
counts_mean = counts_all.mean(dim="track", skipna=True)

# Build scatter inputs: every (track, x) point
x_vals = counts_all["x"].values
x_scatter = np.broadcast_to(x_vals, counts_all.shape).ravel()
y_scatter = counts_all.values.ravel()

fig, ax = plt.subplots(figsize=(10, 5))

# All track counts points as cyan dots
ax.scatter(x_scatter, y_scatter, s=8, c="cyan", alpha=0.6, edgecolors="none")

# Mean counts as thick cyan solid line
ax.plot(counts_mean["x"].values, counts_mean.values, color="b", linewidth=3.5, linestyle="-")

ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
ax.set_xlim(-10e3, 300e3)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("counts (m)")
ax.set_title("All track counts points and mean counts")
ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()

ds.close()
