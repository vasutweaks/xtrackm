import numpy as np
import xarray as xr

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

ekman_velocities = calculate_ekman_velocities(dsu, dsv, z)
ekman_velocities.to_netcdf("ekman_velocities_15m_clim.nc", format="NETCDF4")
# u_ekman1 = ekman_velocities.u_ekman.isel(month=7)
# u_ekman1.plot()
# ekman_velocities.to_netcdf("ekman_velocities.nc")
