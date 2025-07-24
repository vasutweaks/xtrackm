import cmaps
import cmocean as cmo
import xarray as xr
import matplotlib.pyplot as plt

cmap1 = cmo.cm.topo
cmap2 = cmo.cm.thermal
cmap1 = cmo.cm.topo
cmap3 = cmaps.BlAqGrYeOrReVi200
cmap4 = cmo.cm.phase
cmap_rmse = "YlGn"

width, height = 10, 10
WBOB = (78.0, 87.0, 8.0, 21.0)
TN_RADAR = (79.6, 81.8, 10.5, 12.9)

dsr = xr.open_dataset("/home/srinivasu/allData/radar/TNCodar_daily.nc")
print(dsr)
dsr = dsr.rename(
    {"XAXS": "longitude", "YAXS": "latitude", "ZAXS": "lev", "TAXIS1D": "time"}
)
print(dsr)
u_radar = 0.01 * dsr.U_RADAR.isel(lev=0, drop=True)
u_radar_m = u_radar.resample(time="1M").mean()
v_radar = 0.01 * dsr.V_RADAR.isel(lev=0, drop=True)
v_radar_m = v_radar.resample(time="1M").mean()
print(u_radar)

sizes = v_radar.sizes
ln = sizes["time"]
ngp_u = 100*(v_radar.count(dim="time")/ln)
fig, ax = plt.subplots(1,1, figsize=(6,4), layout="constrained")
c = ngp_u.plot(ax=ax, cmap=cmap3, add_colorbar=False)
cb = plt.colorbar(c)
cb.set_label("% of valid data points")
plt.savefig(f"TNradar_valid_points.png")
plt.show()
