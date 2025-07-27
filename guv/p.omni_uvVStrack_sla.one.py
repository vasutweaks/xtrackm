import cmocean as cmo
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from tools_xtrackm import *
import sys

cmap1 = cmo.cm.diff
sat = "S3A"
sat = "ERS1+ERS2+ENV+SRL"
sat = "TP+J1+J2+J3+S6A"
d = 1.5
height, width = 9, 9


def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate great-circle distance between (lon1, lat1)
    and arrays (lon2, lat2) in kilometers."""
    R = 6371  # Radius of Earth in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2)**2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))


def find_n_closest_points(df, lon0, lat0, n=4):
    """Return the n rows in df closest to (lon0, lat0) based on
    lons_inter/lats_inter."""
    distances = haversine_distance(lon0, lat0, df["lons_inter"].values,
                                   df["lats_inter"].values)
    df = df.copy()
    df["distance_km"] = distances
    return df.nsmallest(n, "distance_km")


omni_id = "BD09"
lon_omni, lat_omni = omni_d[omni_id]
df = pd.read_csv(f"tracks_intersections_{sat}_1.csv")
df_4 = find_n_closest_points(df, lon_omni, lat_omni, n=4)
print(omni_id)
print(df_4)

# sys.exit()
track_self = df_4["track_self"].apply(lambda x: str(x)).values
track_other = df_4["track_other"].apply(lambda x: str(x)).values
lons_inter = df_4["lons_inter"].values
lats_inter = df_4["lats_inter"].values
x_from_coast_self = df_4["x_from_coast_self"].values
x_from_coast_other = df_4["x_from_coast_other"].values
angle_acute = df_4["angle_acute"].values
angle_obtuse = df_4["angle_obtuse"].values
dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset

for i in range(len(df_4)):
    track_number_self = track_self[i]
    track_number_other = track_other[i]
    f_self = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_self.zfill(3)}.nc"
    f_other = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{track_number_other.zfill(3)}.nc"
    lon1 = lons_inter[i]
    lat1 = lats_inter[i]
    x_from_coast_self1 = x_from_coast_self[i]
    x_from_coast_other1 = x_from_coast_other[i]
    angle_acute1 = angle_acute[i]
    angle_obtuse1 = angle_obtuse[i]
    ds_self = xr.open_dataset(f_self, decode_times=False)
    sla_self = track_dist_time_asn(ds_self, var_str="sla", units_in="m")
    ds_other = xr.open_dataset(f_other, engine="h5netcdf", decode_times=False)
    sla_other = track_dist_time_asn(ds_other, var_str="sla", units_in="m")
    lons_track_self = ds_self.lon.values
    lats_track_self = ds_self.lat.values
    lon_coast_self = lons_track_self[-1]  # this on coast
    lat_coast_self = lats_track_self[-1]  # this on coast
    lon_equat_self = lons_track_self[0]  # this on equator
    lat_equat_self = lats_track_self[0]  # this on equator
    slope_self = (lats_track_self[-1] - lats_track_self[0]) / (
        lons_track_self[-1] - lons_track_self[0])
    angle_self = np.rad2deg(np.arctan(slope_self))
    track_path_self = sg.LineString(zip(lons_track_self, lats_track_self))
    lons_track_other = ds_other.lon.values
    lats_track_other = ds_other.lat.values
    lon_equat_other = lons_track_other[0]
    lat_equat_other = lats_track_other[0]
    lon_coast_other = lons_track_other[-1]
    lat_coast_other = lats_track_other[-1]
    slope_other = (lat_coast_other - lat_equat_other) / (lon_coast_other - lon_equat_other)
    angle_other = np.rad2deg(np.arctan(slope_other))
    track_path_other = sg.LineString(zip(lons_track_other, lats_track_other))
    # if track_path_self.intersects(track_path_other):
    sla_self_at = sla_self.sel(x=x_from_coast_self1,
                               method="nearest",
                               drop=True)
    sla_other_at = sla_other.sel(x=x_from_coast_other1,
                                 method="nearest",
                                 drop=True)
    sla_other_at = sla_other_at.interp(time=sla_self_at.time)
    # other_times.append((track_number_other, abs(dist_time)))
    fig1, ax1 = plt.subplots(1,
                            1,
                            figsize=(width, height),
                            layout="constrained")
    sla_self_at.plot(ax=ax1, label=f"pass {track_number_self}")
    sla_other_at.plot(ax=ax1, label=f"pass {track_number_other}")
    # text coordinate of intersection point
    plt.text(
        0.15,
        0.95,
        f"intersection point ({lon1:.2f}, {lat1:.2f})",
        transform=ax1.transAxes,
        fontsize=14,
        fontweight="bold",
    )
    plt.legend()
    fig2, ax2 = plt.subplots(
        1,
        1,
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(width, height),
        layout="constrained",
    )
    dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(ETOPO05_Y=slice(
        *TRACKS_REG[2:])).plot(ax=ax2,
                               add_colorbar=False,
                               add_labels=False,
                               cmap=cmap1)
    decorate_axis(ax2, "", *TRACKS_REG)
    ax2.grid()
    ax2.plot(
        lon_omni,
        lat_omni,
        c="r",
        marker="o",
        markersize=6,
        markerfacecolor='red',
        markeredgecolor='black',
        markeredgewidth=2
    )
    # plot plat form code of omni at lon_omni, lat_omni
    plt.text(lon_omni, lat_omni, f"{omni_id}", fontsize=10, color="k") 
    ax2.scatter(
        lons_track_self,
        lats_track_self,
        marker=".",
        color="c",
        s=4,
    )
    ax2.scatter(
        lons_track_other,
        lats_track_other,
        marker=".",
        color="c",
        s=4,
    )
    lonm, latm = get_point_at_distance(lon_equat_self, lat_equat_self,
                                       lon_coast_self, lat_coast_self, d)
    if is_within_region(lonm, latm, *TRACKS_REG):
        plt.text(
            lonm,
            latm,
            s=track_number_self,
            fontsize=14,
            rotation=angle_self,
            color="w",
        )
    lonm, latm = get_point_at_distance(lon_equat_other, lat_equat_other, lon_coast_other, lat_coast_other, d)
    if is_within_region(lonm, latm, *TRACKS_REG):
        plt.text(
            lonm,
            latm,
            s=track_number_other,
            fontsize=14,
            rotation=angle_other,
            color="w",
        )
    plt.show()
    plt.close("all")
    break

dse.close()
