import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tools_xtrackm import decorate_axis

ekman_dir = f"/home/srinivasu/slnew/xtrackd/computed/"
oscar_dir = f"/home/srinivasu/allData/oscar_total/data/"

f_ekman = f"{ekman_dir}/2020_ekman_uv_ccmp3.0.nc"
f_oscar = f"{oscar_dir}/alltime_oscar_currents_interim_0.25deg_nio.nc"

ds_ekman = xr.open_dataset(f_ekman)
ds_oscar = xr.open_dataset(f_oscar)

# ccmp_ek_u = ds_ekman.ekman_u
# oscar_ek_u = ds_oscar.u - ds_oscar.ug
ds_ekman_interp = ds_ekman.interp(lon=ds_oscar.lon, lat=ds_oscar.lat)

ccmp_ek_u = ds_ekman_interp.ekman_u
oscar_ek_u = ds_oscar.u - ds_oscar.ug
oscar_ek_u = oscar_ek_u.transpose('time', 'latitude', 'longitude')

# 2x1 subplots of with Plate projection

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), subplot_kw={'projection': ccrs.PlateCarree()})

cbar_kwargs = {
    'orientation': 'horizontal',
    'shrink': 0.8,  # Adjust the size of the color bar
    'label': 'Ekman U-velocity (m/s)'
}

ccmp_ek_u.mean(dim='time').plot(ax=ax1, transform=ccrs.PlateCarree(), cbar_kwargs=cbar_kwargs)
decorate_axis(ax1, title1="", xsta=42, xend=99, ysta=0, yend=26, step=5)
oscar_ek_u.mean(dim='time').plot(ax=ax2, transform=ccrs.PlateCarree(), cbar_kwargs=cbar_kwargs)
decorate_axis(ax2, title1="", xsta=42, xend=99, ysta=0, yend=26, step=5)

plt.show()
