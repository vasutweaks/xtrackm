import os
import glob
import xarray as xr
import numpy as np

data_loc = f"/home/srinivasu/allData/drifter1/"
chunks = ["1_5000", "5001_10000", "10001_15000", "15001_current"]
chunk_time_dict = {}
for chunk in chunks:
    start_times = []
    end_times = []
    for fd in sorted(
            glob.glob(f"{data_loc}/netcdf_{chunk}/"
                      f"track_reg/drifter_6h_*.nc")):
        basename1 = os.path.basename(fd)
        ds_d = xr.open_dataset(fd, drop_variables=["WMO"])
        drift_tsta_o1, drift_tend_o1 = (
            ds_d.start_date.values[0],
            ds_d.end_date.values[0],
        )
        # print(type(drift_tsta_o1), type(drift_tend_o1))
        start_times.append(drift_tsta_o1)
        end_times.append(drift_tend_o1)
    chunk_time_dict[chunk] = (np.min(start_times), np.max(end_times))
    print(f"min of start times for {chunk} is {np.min(start_times)}")
    print(f"max of end times for {chunk} is {np.max(end_times)}")

print(chunk_time_dict)
