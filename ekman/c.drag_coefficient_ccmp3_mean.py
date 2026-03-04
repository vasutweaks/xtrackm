import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

dsw = xr.open_dataset("/home/srinivasu/allData/ncep2/wspd.10m.mon.mean.nc")
print(dsw)
# speed1 = dsw.wspd.sel(time=slice("01-01-1992", "2024-01-01"))
speed1m = dsw.wspd.mean(dim="time")
speed2m = speed1m.sel(lon=slice(30, 120))
speed3m = speed2m.sel(lat=slice(20, -20))

ds = xr.open_dataset(
    f"/home/srinivasu/allData/ccmp3.0_lt/2003_CCMP_Wind_Analysis_io_daily.nc",
)
print(ds)
ds = ds.rename({"longitude": "lon", "latitude": "lat"})
print(ds)
uwnd1 = ds.uwnd
vwnd1 = ds.vwnd
speed1 = (uwnd1**2.0 + vwnd1**2.0) ** 0.5

# def compute_drag_coefficient(U10):
#     """
#     Compute the drag coefficient CD based on the wind speed U10 using the given piecewise formula.
#     
#     Parameters:
#     U10 (xarray.DataArray): 2D DataArray of wind speed at 10 meters (m/s)
#     
#     Returns:
#     xarray.DataArray: 2D DataArray of drag coefficient CD
#     """
#     # Initialize CD array with the same shape as U10
#     CD = xr.full_like(U10, np.nan)
#     
#     # Apply the first formula for 3 <= U10 <= 6 m/s
#     mask_3_to_6 = (U10 >= 3) & (U10 <= 6)
#     CD = 0.001 * CD.where(~mask_3_to_6, 0.29 + (3.1 / U10) + (7.7 / U10**2))
#     
#     # Apply the second formula for 6 < U10 <= 26 m/s
#     mask_6_to_26 = (U10 > 6) & (U10 <= 26)
#     CD = 0.001 * CD.where(~mask_6_to_26, 0.60 + (0.071 * U10))
#     
#     return CD

def compute_drag_coefficient_ekmanlib(w):
    """
    https://github.com/Konfersi-Indonesia/ekmanlib/blob/main/ekmanlib/ekmanlib.py
    Computes non linear drag coefficient from Trenberth et al. (1990) as a function of wind speed
    Formula and a non-linear drag coefficient (cd) based on
    Large and Pond (1981)[2], modified for low wind speeds (Trenberth et
    al., 1990)[3]

    [1] A.E. Gill, 1982, Atmosphere-Ocean Dynamics, Academy Press.
    [2] W.G. Large & S. Pond., 1981,Open Ocean Measurements in Moderate
    to Strong Winds, J. Physical Oceanography, v11, p324-336.
    [3] K.E. Trenberth, W.G. Large & J.G. Olson, 1990, The Mean Annual
    Cycle in Global Ocean Wind Stress, J. Physical Oceanography, v20,
    p1742-1760.
    """
    C_N = xr.full_like(w, np.nan)
    C_N = xr.where(w > 10, 0.49 + 0.065 * w, C_N)
    C_N = xr.where((w > 3) & (w <= 10), 1.14, C_N)
    C_N = xr.where((w > 1) & (w <= 3), 0.62 + (1.56/w), C_N)
    C_N = xr.where(w <= 1, 2.18e-3, C_N)
    C_N = 0.001 * C_N
    return C_N

# def compute_drag_coefficient_ekmanlib(w):
#     """Convert wind speed (u,v) to wind stress (Tx,Ty).
#
#     It uses either wind-dependent or constant drag.
#
#     Args:
#         u, v: Wind vector components (m/s), 2d or 3d (for time series).
#         rho_air: Density of air (1.22 kg/m^3).
#         cd: Non-dimensional drag (wind-speed dependent).
#             For constant drag use cd=1.5e-3.
#     Notes:
#         Function to compute wind stress from wind field data is based on Gill,
#         (1982)[1]. Formula and a non-linear drag coefficient (cd) based on
#         Large and Pond (1981)[2], modified for low wind speeds (Trenberth et
#         al., 1990)[3]
#
#         [1] A.E. Gill, 1982, Atmosphere-Ocean Dynamics, Academy Press.
#         [2] W.G. Large & S. Pond., 1981,Open Ocean Measurements in Moderate
#         to Strong Winds, J. Physical Oceanography, v11, p324-336.
#         [3] K.E. Trenberth, W.G. Large & J.G. Olson, 1990, The Mean Annual
#         Cycle in Global Ocean Wind Stress, J. Physical Oceanography, v20,
#         p1742-1760.
#     """
#     # wind-dependent drag
#     cond1 = (w<=1)
#     cond2 = (w>1) & (w<=3)
#     cond3 = (w>3) & (w<10)
#     cond4 = (w>=10)
#     cd = np.zeros_like(w)
#     cd[cond1] = 2.18e-3 
#     cd[cond2] = (0.62 + 1.56/w[cond2]) * 1e-3
#     cd[cond3] = 1.14e-3
#     cd[cond4] = (0.49 + 0.065*w[cond4]) * 1e-3
#     return cd

# CD = compute_drag_coefficient(speed1)
fig, ax = plt.subplots(figsize=(12, 8))
CD = compute_drag_coefficient_ekmanlib(speed1)
print(CD)
# CD.isel(time=1).plot(vmax=0.002, cmap="tab20c")
# CD.mean(dim="time").plot(vmin=0.00111, vmax=0.00116, cmap="tab20c", ax=ax)
CD.mean(dim="time").plot(cmap="tab20c", ax=ax)
CS = speed1.mean(dim="time").plot.contour(ax=ax, colors="k")
ax.clabel(CS, inline=1, fontsize=10)
plt.show()
