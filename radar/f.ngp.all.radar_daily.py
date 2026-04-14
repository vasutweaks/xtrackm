import cmaps
import cmocean as cmo
import xarray as xr
import matplotlib.pyplot as plt

cmap1 = cmo.cm.topo
cmap4 = cmo.cm.phase
cmap_rmse = "YlGn"
cmap3 = cmo.cm.thermal
cmap3 = cmaps.BlAqGrYeOrReVi200

WBOB = (78.0, 87.0, 8.0, 21.0)

RADARS = [
    ("AP", "/home/srinivasu/allData/radar/APCodar_daily.nc", (80.5, 83.5, 14.2, 16.5)),
    ("OR", "/home/srinivasu/allData/radar/ORCodar_daily.nc", (84.5, 87.0, 17.5, 20.0)),
    ("TN", "/home/srinivasu/allData/radar/TNCodar_daily.nc", (79.6, 81.8, 10.5, 12.9)),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
axes = axes.ravel()
mappable = None

for i, (name, fpath, bbox) in enumerate(RADARS):
    dsr = xr.open_dataset(fpath).rename(
        {"XAXS": "longitude", "YAXS": "latitude", "ZAXS": "lev", "TAXIS1D": "time"}
    )
    v_radar = 0.01 * dsr.U_RADAR.isel(lev=0, drop=True)
    ln = v_radar.sizes["time"]
    ngp = 100 * (v_radar.count(dim="time") / ln)

    ax = axes[i]
    mappable = ngp.plot(
        ax=ax,
        cmap=cmap3,
        vmin=0,
        vmax=100,
        add_colorbar=False,
    )
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_title(f"{name} radar")

axes[-1].set_visible(False)

cb = fig.colorbar(mappable, ax=axes.tolist(), shrink=0.95)
cb.set_label("% of valid data points")

plt.savefig("all_radar_valid_points_2x2.png", dpi=300)
plt.show()
