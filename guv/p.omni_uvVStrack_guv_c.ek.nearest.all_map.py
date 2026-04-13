import os

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap
from tools_xtrackm import *


def build_unique_str(sat, row):
    track_self = track_to_str(row["track_self"])
    track_other = track_to_str(row["track_other"])
    return f"{sat}_{track_self}_{track_other}"


def track_to_str(track):
    track_f = float(track)
    if track_f.is_integer():
        return str(int(track_f))
    return str(track)


def filter_omni_uv_72h(omni_u, omni_v):
    omni_u_filt = butter_lowpass_1d(omni_u, cutoff_days=3, order=4)
    omni_v_filt = butter_lowpass_1d(omni_v, cutoff_days=3, order=4)
    return omni_u_filt, omni_v_filt


cmap1 = ListedColormap(["white"])
d = 1.5
height, width = 9, 9
frac1 = "0.05"
method = "loess"
BOB_REG = (78.0, 95.0, 4.0, 24.0)
AS_REG = (54.0, 78.0, 4.0, 24.0)

sat_marker_dict = {
    "GFO": "*",
    "TP+J1+J2+J3+S6A": "o",
    "S3A": "^",
    "S3B": "^",
    "TPN+J1N+J2N+J3N": "o",
    "ERS1+ERS2+ENV+SRL": "x",
    "HY2A": "s",
    "HY2B": "s",
}


dse = xr.open_dataset("~/allData/topo/etopo5.cdf")
rose_cut = dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(
    ETOPO05_Y=slice(*TRACKS_REG[2:])
)
omni_data_dir = "/home/srinivasu/allData/MB_cleaned/raju_omni_clean"
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")
rose = dse.ROSE
guv_dir = f"/home/srinivasu/xtrackm/guv/guv_at_intersections_{method}"

df_all_list = []
for sat in sats_new:
    df_all_list.append(pd.read_csv(f"tracks_intersections_{sat}_1.csv"))
df_all = pd.concat(df_all_list, ignore_index=True)
satellites = df_all["sat"].unique()
colors = plt.cm.get_cmap("tab10", len(satellites))

for omni_id in omni_d.keys():
    lon_omni, lat_omni = omni_d[omni_id]
    print(omni_id, lon_omni, lat_omni)
    omni_file = f"{omni_data_dir}/{omni_id}_current_sst.nc"
    print(omni_file)
    if not os.path.exists(omni_file):
        print(f"File not found: {omni_file}")
        continue
    if lat_omni < 5:
        continue

    ds_omni = xr.open_dataset(omni_file)
    print(ds_omni)
    df_4 = find_n_closest_points(df_all, lon_omni, lat_omni, n=4)
    lons_df4 = df_4["lons_inter"].to_numpy()
    lats_df4 = df_4["lats_inter"].to_numpy()
    sats_df4 = df_4["sat"].tolist()
    print(sats_df4)
    map_reg = BOB_REG if lon_omni >= 80.0 else AS_REG
    rose_cut = rose.sel(ETOPO05_X=slice(*map_reg[:2])).sel(
        ETOPO05_Y=slice(*map_reg[2:])
    )
    fig = plt.figure(figsize=(9, 9))
    ax2 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    rose_cut.plot(ax=ax2, add_colorbar=False, add_labels=False, cmap=cmap1)
    decorate_axis(ax2, "", *map_reg, step=4)
    ax2.set_extent([*map_reg], crs=ccrs.PlateCarree())
    ax2.coastlines()
    ax2.grid()
    for i, sat_all in enumerate(satellites):
        sat_data = df_all[df_all["sat"] == sat_all]
        ax2.scatter(
            sat_data["lons_inter"],
            sat_data["lats_inter"],
            color=colors(i),
            s=12,
            marker=sat_marker_dict.get(sat_all, "."),
            transform=ccrs.PlateCarree(),
            zorder=1,
        )
    ax2.plot(
        lon_omni,
        lat_omni,
        c="r",
        marker="o",
        markersize=4,
        markerfacecolor="red",
        markeredgecolor="red",
        markeredgewidth=2,
    )
    ax2.scatter(
        lons_df4,
        lats_df4,
        marker="o",
        s=75,
        facecolors="none",
        edgecolors="orange",
        linewidths=2.0,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )
    plt.text(lon_omni, lat_omni, f"{omni_id}", fontsize=10, color="k")

    plt.show()
    plt.close("all")

dse.close()
