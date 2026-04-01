import pandas as pd
import xarray as xr
from pathlib import Path
from namesm import sats_new
import matplotlib.pyplot as plt

ekman_dir = "/home/srinivasu/allData/ccmp31/"
out_dir = Path("/home/srinivasu/xtrackm/ekman/ekman_at_intersections_cd/")
out_dir.mkdir(parents=True, exist_ok=True)

df_all_list = [pd.read_csv(f"tracks_intersections_{sat}_1.csv") for sat in sats_new]
inter_df = pd.concat(df_all_list, ignore_index=True)

for z_in in ["0", "10", "15"]:
    ds_ekman = xr.open_dataset(
        f"{ekman_dir}/ekman_velocities_{z_in}m_all_years_daily_cd.nc",
        chunks={"time": 365, "lat": 50, "lon": 50}
    )
    u_ekman = ds_ekman["u_ekman"]
    v_ekman = ds_ekman["v_ekman"]
    for i, r in inter_df.iterrows():
        sat = r["sat"]
        track_number_self = str(r["track_self"])
        track_number_other = str(r["track_other"])
        lons_inter1 = r["lons_inter"]
        lats_inter1 = r["lats_inter"]
        unique_str = f"{sat}_{track_number_self}_{track_number_other}"
        print(unique_str, z_in)
        u_at = u_ekman.sel(
            latitude=lats_inter1, longitude=lons_inter1, method="nearest"
        )
        v_at = v_ekman.sel(
            latitude=lats_inter1, longitude=lons_inter1, method="nearest"
        )
        # u_at.plot()
        # plt.show()
        ds_out = xr.Dataset(
            {
                "u_at": u_at,
                "v_at": v_at,
            }
        )
        ds_out.to_netcdf(
            f"{out_dir}/{unique_str}_ekman_{z_in}m_at_intersections_cd.nc", format="NETCDF4"
        )
