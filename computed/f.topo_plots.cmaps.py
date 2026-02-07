import xarray as xr
from tools_xtrackm import *
import matplotlib as mpl

dse = xr.open_dataset("/home/srinivasu/allData/topo/etopo5.cdf")
topo = dse.ROSE.sel(ETOPO05_X=slice(*NIO[:2])).sel(ETOPO05_Y=slice(*NIO[2:]))

for cmap in mpl.colormaps:
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    topo.plot(ax=ax1, add_colorbar=False, add_labels=False, cmap=cmap)
    decorate_axis(ax1, "", *NIO, step=8)

    info = f"name of cmap: {cmap}"
    plt.text(
        0.15,
        0.90,
        info,
        transform=ax1.transAxes,
        fontsize=14,
        fontweight="bold",
        color="w",
    )

    plt.savefig(f"topo_cmap_plots/topo.{cmap}.png", dpi=300)
    # plt.show()
    plt.close(fig)

dse.close()
