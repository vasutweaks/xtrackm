import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature

dsu = xr.open_dataset(f"ccmp_v31_daily_iod_u_mon_clim.nc")
dsv = xr.open_dataset(f"ccmp_v31_daily_iod_v_mon_clim.nc")


def calculate_ekman_velocities(dsu, dsv, z, A=0.1, cd=1.5e-3, rho_air=1.225, rho_water=1025.0):
    """
    Calculates the upper ocean Ekman velocities from CCMP 10m wind fields.
    Parameters:
    ds : xarray.Dataset
        Dataset containing CCMP variables 'uwnd', 'vwnd', and 'latitude'.
    z_levels : array-like
        Depths (in meters, should be negative as z=0 is the surface and z -> -inf).
    A : float
        Eddy viscosity in m^2/s. Default is 0.1.
    cd : float
        Drag coefficient for bulk wind stress formula.
    rho_air : float
        Air density in kg/m^3.
    rho_water : float
        Ocean reference density in kg/m^3.
    Returns:
    xarray.Dataset
        Dataset containing the Ekman velocity components 'u_ekman' and 'v_ekman'.
    """
    # 1. Extract 10m wind components from CCMP data
    u10 = dsu['uwnd']
    v10 = dsv['vwnd']
    lat = dsu['latitude']
    # 2. Calculate kinematic wind stress (tau_tilde)
    # NOTE: The bulk aerodynamic formula to convert 10m winds to surface stress 
    # uses information from outside the provided sources and should be independently verified.
    wind_speed = np.sqrt(u10**2 + v10**2)
    tau_x = rho_air * cd * wind_speed * u10
    tau_y = rho_air * cd * wind_speed * v10
    # Kinematic wind stress is stress divided by reference density
    tau_tilde_x = tau_x / rho_water
    tau_tilde_y = tau_y / rho_water
    # 3. Calculate Coriolis parameter (f) and mask the equator to avoid division by zero
    omega = 7.2921e-5 # Earth's rotation rate in rad/s
    f = 2 * omega * np.sin(np.deg2rad(lat))
    f = f.where(np.abs(lat) > 2.0) # Masking out 2 degrees near the equator
    # 4. Calculate Ekman layer depth (d)
    # The absolute value of f is used to ensure the depth is real and positive.
    d = np.sqrt(2 * A / np.abs(f))
    # 5. Calculate Ekman velocities for each z level
    decay = np.exp(z / d)
    phase_x = np.cos(z / d - np.pi / 4)
    phase_y = np.sin(z / d - np.pi / 4)
        # Ekman velocity equations
    u_E = (np.sqrt(2) / (f * d)) * decay * (tau_tilde_x * phase_x - tau_tilde_y * phase_y)
    v_E = (np.sqrt(2) / (f * d)) * decay * (tau_tilde_x * phase_y + tau_tilde_y * phase_x)
    return xr.Dataset({'u_ekman': u_E, 'v_ekman': v_E})

z = -15
# z = 0
# z = -30

ekman_velocities = calculate_ekman_velocities(dsu, dsv, z)
z_out = str(abs(z))
f_out = f"ekman_velocities_{z_out}m_clim.nc"
ekman_velocities.to_netcdf(f_out, format="NETCDF4")

INFILE = f_out
OUTFIG = (
    f"/home/srinivasu/allData/ccmp31/p.u_E.{z_out}m_monsoon_panel.png"
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
    cbar.set_label("u_E (m s$^{-1}$)")
    # fig.suptitle("OSCAR - MSLA Ageostrophic U: Seasonal Means", y=0.98)
    fig.savefig(OUTFIG, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Wrote figure: {OUTFIG}")

if __name__ == "__main__":
    main()
