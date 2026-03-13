import pandas as pd
import xarray as xr
from pathlib import Path
from namesm import sats_new
import matplotlib.pyplot as plt

msla_uv_dir = "/home/srinivasu/allData/aviso/msla_uv/"
out_dir = Path("/home/srinivasu/xtrackm/guv/msla_uv_at_intersections/")
out_dir.mkdir(parents=True, exist_ok=True)

ds_msla_u = xr.open_dataset(
    f"{msla_uv_dir}/allyear_dt_global_allsat_msla_u.nc",
    chunks={"time": 365, "lat": 50, "lon": 50}
)
ds_msla_v = xr.open_dataset(
    f"{msla_uv_dir}/allyear_dt_global_allsat_msla_v.nc",
    chunks={"time": 365, "lat": 50, "lon": 50}
)

u_msla = ds_msla_u["u"]
v_msla = ds_msla_v["v"]

df_all_list = [pd.read_csv(f"tracks_intersections_{sat}_1.csv") for sat in sats_new]
inter_df = pd.concat(df_all_list, ignore_index=True)

for i, r in inter_df.iterrows():
    sat = r["sat"]
    if sat != "TP+J1+J2+J3+S6A":
        continue
    track_number_self = str(r["track_self"])
    track_number_other = str(r["track_other"])
    lons_inter1 = r["lons_inter"]
    lats_inter1 = r["lats_inter"]
    unique_str = f"{sat}_{track_number_self}_{track_number_other}"
    print(unique_str)
    u_at = u_msla.sel(
        latitude=lats_inter1, longitude=lons_inter1, method="nearest"
    )
    v_at = v_msla.sel(
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
        f"{out_dir}/{unique_str}_msla_uv_at_intersections.nc", format="NETCDF4"
    )
