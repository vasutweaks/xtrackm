# rama location plotted only for reference
from glob import glob

import cartopy.crs as ccrs
import cmocean as cmo
import matplotlib.pyplot as plt
import xarray as xr
from tools_xtrackm import *

cmap = "Spectral"
cmap = "PuBuGn"
cmap = "YlGnBu"
cmap = "YlGn"
cmap_r = plt.cm.get_cmap(cmap)
cmap1 = cmo.cm.topo
cmap1 = cmo.cm.topo
cmap1 = cmo.cm.diff

height, width = 9, 14
d = 1.5

sats = ["ERS1+ERS2+ENV+SRL"]
sats = ["GFO"]
sat = sats[0]

lons_inter = []
lats_inter = []
track_self = []
track_other = []

rama_loc = "/home/srinivasu/allData/rama/"
var_str = "cur"
var_str1 = "cur"
dt0 = datetime.strptime("1950-01-01", "%Y-%m-%d")
rama_id = "15n90e"
lon_rama, lat_rama = rama_d[rama_id]

dse = xr.open_dataset("~/allData/topo/etopo5.cdf")  # open etopo dataset
fig, ax1 = plt.subplots(
    1,
    1,
    subplot_kw={"projection": ccrs.PlateCarree()},
    figsize=(width, height),
    layout="constrained",
)
dse.ROSE.sel(ETOPO05_X=slice(*TRACKS_REG[:2])).sel(ETOPO05_Y=slice(
    *TRACKS_REG[2:])).plot(ax=ax1,
                           add_colorbar=False,
                           add_labels=False,
                           cmap=cmap1)
decorate_axis(ax1, "", *TRACKS_REG)
ax1.grid()
ax1.plot(
    lon_rama,
    lat_rama,
    c="y",
    marker="D",
    markersize=15,
)
# 15n90e 129.0 14.0

for f_self in sorted(glob.glob(f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc")):
    ds_self = xr.open_dataset(f_self, decode_times=False)
    track_number = ds_self.Pass
    print(ds_self)
    sla = ds_self.sla
    sla1 = sla.isel(cycles_numbers=0)
    ln = ds_self["sla"].sizes["points_numbers"]
    ln2 = ln // 2
    lons_track_self = ds_self.lon.values
    lats_track_self = ds_self.lat.values
    lon_coast_self = lons_track_self[-1]  # this on coast
    lat_coast_self = lats_track_self[-1]  # this on coast
    lon_eqaut_self = lons_track_self[0]  # this on equator
    lat_eqaut_self = lats_track_self[0]  # this on equator
    if len(lons_track_self) == 0:
        continue
    slope_self = (lats_track_self[-1] - lats_track_self[0]) / (lons_track_self[-1] - lons_track_self[0])
    angle_self = np.rad2deg(np.arctan(slope_self))
    ax1.scatter(
        lons_track_self,
        lats_track_self,
        marker=".",
        color="c",
        s=4,
    )
    # for plotting the pass number at a slight distance from the track
    # from the middle
    lonm, latm = get_point_at_distance(lon_eqaut_self, lat_eqaut_self,
                                       lon_coast_self,
                                       lat_coast_self, d)
    if is_within_region(lonm, latm, *TRACKS_REG):
        plt.text(
            lonm,
            latm,
            s=track_number,
            fontsize=10,
            rotation=angle_self,
            color="w",
        )
    if slope_self < 0:
        continue
    track_path_self = sg.LineString(zip(lons_track_self, lats_track_self))
    for f_other in sorted(
            glob.glob(f"../data/{sat}_lon_ordered/ctoh.sla.ref.{sat}.nindian.*.nc")):
        # print(f)
        ds_other = xr.open_dataset(f_other,
                                   engine="h5netcdf",
                                   decode_times=False)
        if len(ds_other.points_numbers) == 0:
            continue
        track_number_other = ds_other.Pass
        lons_track_other = ds_other.lon.values
        lats_track_other = ds_other.lat.values
        lon_equat_other = lons_track_other[0]
        lat_equat_other = lats_track_other[0]
        lon_coast_other = lons_track_other[-1]
        lat_coast_other = lats_track_other[-1]
        slope_other = (lat_coast_other - lat_equat_other) / (
            lon_coast_other - lon_equat_other)
        if slope_other > 0:
            continue
        # angle_r_other = math.atan(slope_other)
        # angle_d_other = angle_r_other * (180 / math.pi)
        # define a linestring for the track
        track_path_other = sg.LineString(
            zip(lons_track_other, lats_track_other))
        other_times = []
        if track_path_self.intersects(track_path_other):
            point = track_path_self.intersection(track_path_other)
            print(sat, track_number, track_number_other, point)
            # x_from_coast = distance.distance((lat_coast_self,
            #                                   lon_coast_self),
            #                                  (point.y, point.x)).km
            # x_idx = index_at_x(x, x_from_coast)
            lat_idx = index_at_lat(ds_other, point.y)
            ln = ds_other["sla"].sizes["points_numbers"]
            ln2 = ln // 2
            # dvals_other = ds_other.time.isel(points_numbers=ln2).values
            dvals_self = ds_self.time.isel(points_numbers=lat_idx).values
            # these dates change by weeks or months
            dates_self = [dt0 + timedelta(days=int(d)) for d in dvals_self]
            time0_self = dates_self[0]
            dvals_other = ds_other.time.isel(points_numbers=lat_idx).values
            dates_other = [dt0 + timedelta(days=int(d)) for d in dvals_other]
            time0_other = dates_other[0]
            dist_time = (time0_other - time0_self).days

            ax1.plot(
                point.x,
                point.y,
                c="k",
                marker="x",
                markersize=15,
            )
            other_times.append((track_number_other, abs(dist_time)))
        # plt.pause(0.1)
            track_other.append(track_number_other)
            track_self.append(track_number)
            lons_inter.append(point.x)
            lats_inter.append(point.y)
    break
    if len(other_times) == 0:
        continue
    other_times.sort(key=lambda x: x[1], reverse=True)

df = pd.DataFrame(
    {
        "track_self": track_self,
        "track_other": track_other,
        "lons_inter": lons_inter,
        "lats_inter": lats_inter,
    }
)
df.to_csv(f"tracks_intersections_{sat}1.csv")

plt.show()
plt.close("all")
