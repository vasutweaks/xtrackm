import numpy as np
import xarray as xr
import glob
from matplotlib import pyplot as plt

sat = "TPN+J1N+J2N+J3N"

file_list = sorted(glob.glob(f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc"))
for f in file_list:
    ds = xr.open_dataset(f)
    sla = ds["sla"]
    sla.plot()
    plt.show()
