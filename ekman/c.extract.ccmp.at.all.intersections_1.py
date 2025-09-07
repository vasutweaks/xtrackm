import pandas as pd
import xarray as xr
from pathlib import Path

# ----------------------------------------------------------------------
# 1. Open dataset lazily with dask
# ----------------------------------------------------------------------
ekman_dir = "/home/srinivasu/slnew/xtrackd/computed/"
out_dir = Path("/home/srinivasu/xtrackm/ekman/ekman_at_intersections/")
out_dir.mkdir(parents=True, exist_ok=True)

ds_ekman = xr.open_dataset(
    f"{ekman_dir}/full_series_ekman_uv_ccmp3.0_cdo.nc",
    chunks={"time": 365, "lat": 50, "lon": 50}
)
u_ekman = ds_ekman["ekman_u"]
v_ekman = ds_ekman["ekman_v"]

# ----------------------------------------------------------------------
# 2. Load intersection metadata
# ----------------------------------------------------------------------
df_all_list = [pd.read_csv(f"tracks_intersections_{sat}_1.csv") for sat in sats_new]
df_all = pd.concat(df_all_list, ignore_index=True)

# Choose a satellite group to extract
sat_here = "TP+J1+J2+J3+S6A"
df_here = df_all[df_all["sat"] == sat_here]

# ----------------------------------------------------------------------
# 3. Group by (track_self, track_other, sat) and extract each once
# ----------------------------------------------------------------------
for (ts, to, sat), df_group in df_here.groupby(["track_self", "track_other", "sat"]):

    # Prepare coordinate arrays for this group
    lons = xr.DataArray(df_group["lons_inter"].values, dims="points")
    lats = xr.DataArray(df_group["lats_inter"].values, dims="points")

    # Vectorized extraction for this group
    u_at = u_ekman.sel(lon=lons, lat=lats, method="nearest")
    v_at = v_ekman.sel(lon=lons, lat=lats, method="nearest")

    # Attach metadata to coordinates
    u_at = u_at.assign_coords(
        points=df_group.index.values,
        track_self=("points", df_group["track_self"].astype(str).values),
        track_other=("points", df_group["track_other"].astype(str).values),
        sat=("points", df_group["sat"].values),
    )
    v_at = v_at.assign_coords(
        points=df_group.index.values,
        track_self=("points", df_group["track_self"].astype(str).values),
        track_other=("points", df_group["track_other"].astype(str).values),
        sat=("points", df_group["sat"].values),
    )

    # Save one file per group
    ds_out = xr.Dataset({"u": u_at, "v": v_at})
    outfile = out_dir / f"ekman_at_intersection_{sat}_{ts}_{to}.nc"
    ds_out.to_netcdf(outfile)

    print(f"✅ Saved {outfile}")
    break

