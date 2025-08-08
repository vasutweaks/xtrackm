import glob
import os

import xarray as xr
from tools_xtrackd import *

from drifter_tools import *

REG = (43.0, 99, 0, 25.4)
chunk = "15001_current"
sat = "S3A"
track_tsta_o, track_tend_o = get_time_limits_o(sat)
polygon_freq = 1
dse = xr.open_dataset(
    "/home/srinivasu/allData/topo/etopo_60s_io.nc"
)  # open etopo dataset
print(dse)

for fd in sorted(
    glob.glob(
        f"/home/srinivasu/allData/drifter1/netcdf_{chunk}/track_reg/drifter_6h_*.nc"
    )
):
    print(fd)
    basename1 = os.path.basename(fd)
    ds_d = xr.open_dataset(fd, drop_variables=["WMO"])
    byte_id = ds_d.ID.values[0]
    str_id = byte_id.decode("utf-8")
    drift_tsta_o1, drift_tend_o1 = (
        ds_d.start_date.values[0],
        ds_d.end_date.values[0],
    )
    drift_tsta_o, drift_tend_o = pd.to_datetime(drift_tsta_o1), pd.to_datetime(
        drift_tend_o1
    )
    overlap_tsta, overlap_tend = overlap_dates(
        track_tsta_o, track_tend_o, drift_tsta_o, drift_tend_o
    )
    print(drift_tsta_o, drift_tend_o)
    if overlap_tsta is None or overlap_tend is None:
        print(f"no time overlap for track {sat} and drifter {str_id}")
        continue
    # Combine the sampled lon and lat into a polygon
    lons_drift = ds_d.longitude.isel(traj=0).values
    lats_drift = ds_d.latitude.isel(traj=0).values
    lon_sampled = lons_drift[::polygon_freq]
    lat_sampled = lats_drift[::polygon_freq]
    if len(lon_sampled) < 5:
        continue
    polygon_points = [(lon, lat) for lon, lat in zip(lon_sampled, lat_sampled)]
    polygon = Polygon(polygon_points)
    ve = ds_d.ve.isel(traj=0).values
    fig, ax = create_topo_map(dse, xsta=REG[0], xend=REG[1], ysta=REG[2], yend=REG[3], title1="", step=5)
    sc = ax.scatter(
        lons_drift,
        lats_drift,
        c=ve,
        cmap="viridis",
        s=12,
        transform=ccrs.PlateCarree(),
    )
    # add polygon geometry to the plot
    ax.add_geometries(
        [polygon], ccrs.PlateCarree(), facecolor="none", edgecolor="none"
    )
    for ft in sorted(
        glob.glob(
            f"/home/srinivasu/slnew/xtrackd/data/ctoh.sla.ref.{sat}.nindian.*.nc"
        )
    ):
        basename_t = os.path.basename(ft)
        trackn = basename_t.split(".")[5]
        ds_t = xr.open_dataset(ft, decode_times=False, engine="h5netcdf")
        track_number = ds_t.pass_number
        lons_track = ds_t.lon.values
        lats_track = ds_t.lat.values
        if len(lons_track) == 0:
            continue
        lon_coast = lons_track[-1]  # this on coast
        lat_coast = lats_track[-1]  # this on coast
        line_start = (lons_track[0], lats_track[0])
        line_end = (lons_track[-1], lats_track[-1])
        line = sg.LineString([line_start, line_end])
        if line.intersects(polygon):
            ax.scatter(
                lons_track,
                lats_track,
                c="k",
                s=1,
                transform=ccrs.PlateCarree(),
            )
        else:
            print(f"track {trackn} not intersecting")

    plt.savefig(f"pngs_dump/drifter_{str_id}_with_intersecting_tracks.png")
    # close all windows
    plt.close("all")
    plt.show()
