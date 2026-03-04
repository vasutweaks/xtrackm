import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature

INFILE = "/home/srinivasu/allData/ccmp31/ekman_velocities_15m_clim.nc"
OUTFIG = (
    "/home/srinivasu/allData/ccmp31/p.u_E.monsoon_panel.png"
)
VAR = "u_ekman"
# Seasonal definitions for Indian Ocean monsoon cycle.
SEASONS = {
    "Summer Monsoon (JJAS)": [6, 7, 8, 9],
    "Winter Monsoon (DJF)": [12, 1, 2],
    "Intermonsoon-1 (MAM)": [3, 4, 5],
    "Intermonsoon-2 (ON)": [10, 11],
}

MONTH_DAYS = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}


def weighted_season_mean(da_mon_clim: xr.DataArray, months: list[int]) -> xr.DataArray:
    subset = da_mon_clim.sel(month=months)
    weights = xr.DataArray(
        [MONTH_DAYS[m] for m in months],
        coords={"month": months},
        dims=["month"],
    )
    return (subset * weights).sum("month") / weights.sum()


def main() -> None:
    ds = xr.open_dataset(INFILE)
    if VAR not in ds:
        raise KeyError(f"Variable '{VAR}' not found in {INFILE}")
    da = ds[VAR]
    if "month" not in da.dims:
        raise ValueError("Input file must be monthly climatology with 'month' dimension")
    seasonal = {
        name: weighted_season_mean(da, months) for name, months in SEASONS.items()
    }
    all_vals = np.concatenate([seasonal[s].values.ravel() for s in seasonal])
    finite = all_vals[np.isfinite(all_vals)]
    vmax = np.nanpercentile(np.abs(finite), 98) if finite.size else 0.2
    if vmax <= 0:
        vmax = 0.2
    levels = np.linspace(-vmax, vmax, 21)
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 7.5),
        subplot_kw={"projection": proj},
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.01, right=0.98, top=0.98, bottom=0.07, wspace=0.01, hspace=0.01)
    mappable = None
    for ax, title in zip(axes.ravel(), SEASONS.keys()):
        fld = seasonal[title]
        mappable = ax.contourf(
            fld["longitude"],
            fld["latitude"],
            fld,
            levels=levels,
            cmap="RdBu_r",
            extend="both",
            transform=proj,
        )
        ax.coastlines(linewidth=0.8)
        ax.add_feature(cfeature.LAND, facecolor="0.9", zorder=10)
        ax.set_title(title)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
    cbar = fig.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        pad=0.04,
        fraction=0.045,
        shrink=0.82,
        aspect=35,
    )
    cbar.set_label("u_ageostrophic (m s$^{-1}$)")
    # fig.suptitle("OSCAR - MSLA Ageostrophic U: Seasonal Means", y=0.98)
    fig.savefig(OUTFIG, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Wrote figure: {OUTFIG}")

if __name__ == "__main__":
    main()
