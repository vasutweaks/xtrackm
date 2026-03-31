import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Input file produced by your processing script
nc_file = "std_all_S3A_xtrack.nc"

ds = xr.open_dataset(nc_file)
std_all = ds["std_all"]

# Mean across tracks
std_mean = std_all.mean(dim="track", skipna=True)

# Build scatter inputs: every (track, x) point
x_vals = std_all["x"].values
x_scatter = np.broadcast_to(x_vals, std_all.shape).ravel()
y_scatter = std_all.values.ravel()

fig, ax = plt.subplots(figsize=(10, 5))

# All track STD points as cyan dots
ax.scatter(x_scatter, y_scatter, s=8, c="cyan", alpha=0.6, edgecolors="none")

# Mean STD as thick cyan solid line
ax.plot(std_mean["x"].values, std_mean.values, color="b", linewidth=3.5, linestyle="-")

ax.set_xlabel("Distance (m)")
ax.set_ylabel("STD (m)")
ax.set_title("All track STD points and mean STD")
ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()

ds.close()
