import matplotlib.pyplot as plt
import xarray as xr

ds_xtr = xr.open_dataset("std_mean_S3A_xtrack.nc")
ds_l2p = xr.open_dataset("/home/srinivasu/allData/aviso_l2p/s3a/std_mean_S3A_l2p.nc")

std_xtr = ds_xtr["std"]
std_l2p = ds_l2p["std"]

fig, ax1 = plt.subplots()
ax1.plot(std_xtr, label="xtrack", color="red")
ax1.plot(std_l2p, label="l2p", color="blue")
ax1.set_xlabel("std_mean_xtrack")
ax1.set_ylabel("std_mean_l2p")
ax1.set_title("std_mean_xtrack vs std_mean_l2p")
plt.legend()
plt.show()
