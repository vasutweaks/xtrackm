import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Input file produced by your processing script
nc_xtr = "std_all_S3A_xtrack.nc"
nc_l2p = "/home/srinivasu/allData/aviso_l2p/s3a/std_all_S3A_l2p.nc"

ds_xtr = xr.open_dataset(nc_xtr)
ds_l2p = xr.open_dataset(nc_l2p)
std_all_xtr = ds_xtr["std_all"]
std_all_l2p = ds_l2p["std_all"]

# Mean across tracks
std_mean_xtr = std_all_xtr.mean(dim="track", skipna=True)
std_mean_l2p = std_all_l2p.mean(dim="track", skipna=True)

# Build scatter inputs: every (track, x) point
x_vals_xtr = std_all_xtr["x"].values
x_scatter_xtr = np.broadcast_to(x_vals_xtr, std_all_xtr.shape).ravel()
y_scatter_xtr = std_all_xtr.values.ravel()

x_vals_l2p = std_all_l2p["x"].values
x_scatter_l2p = np.broadcast_to(x_vals_l2p, std_all_l2p.shape).ravel()
y_scatter_l2p = std_all_l2p.values.ravel()

fig, ax = plt.subplots(figsize=(10, 5))

# All track std points as cyan dots
ax.scatter(x_scatter_xtr, y_scatter_xtr, s=8, c="cyan", alpha=0.6, edgecolors="none")
ax.scatter(x_scatter_l2p, y_scatter_l2p, s=8, c="red", alpha=0.6, edgecolors="none")

# Mean std as thick cyan solid line
ax.plot(std_mean_xtr["x"].values, std_mean_xtr.values, color="b", linewidth=3.5, linestyle="-")
ax.plot(std_mean_l2p["x"].values, std_mean_l2p.values, color="r", linewidth=3.5, linestyle="-")

ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}"))
ax.set_xlim(-10e3, 300e3)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("std (m)")
ax.set_title("All track std points and mean std")
ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()

ds.close()
