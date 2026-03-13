import xarray as xr
import os
from tools_xtrackm import *
import numpy as np

method = "odo"


def compute_geostrophy_gc_from_slope(ds, slope_smooth):
    lons_track_rev = ds.lon.values[::-1]
    lats_track_rev = ds.lat.values[::-1]
    if len(lons_track_rev) == 0:
        return None
    # Add minimum length check
    if len(lons_track_rev) < 3:
        print("Warning: Insufficient data points for gradient calculation")
        return None
    f1 = cor_p(lats_track_rev)
    f2 = f1.values
    f3 = np.where(lats_track_rev > 2, f2, np.nan)
    # 2s-2n band equatorial band is masked according to Sudre 2013
    g = grav(lats_track_rev)  # ms-1
    g1 = g.values
    ratio = g1 / f3
    ratio1 = ratio[:, np.newaxis]
    dhbydx = slope_smooth
    gc = ratio1 * dhbydx
    # Add metadata
    gc.attrs["units"] = "m/s"
    gc.attrs["long_name"] = "Cross-track geostrophic current"
    gc.attrs["description"] = "Geostrophic current perpendicular to satellite track"
    return gc


def track_dist_time_asn_midx_da(ds, da_smooth, units_in="m"):
    # dists = compute_dist(ds, units="km")  # to convert to meters
    dists = compute_dist(ds, units=units_in)  # to convert to meters
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lons_track_rev = lons_track[::-1]
    lats_track_rev = lats_track[::-1]
    dt0 = datetime.strptime("1950-01-01", "%Y-%m-%d")
    dists_rev = dists[::-1]
    ln = ds["sla"].sizes["points_numbers"]
    ln2 = ln // 2
    dvals = ds.time.isel(points_numbers=ln2).values
    try:
        dates = [dt0 + timedelta(days=int(d)) for d in dvals]
    except ValueError:
        dvals = pd.Series(dvals).ffill().bfill()
        dates = [dt0 + timedelta(days=int(d)) for d in dvals]
    # now sla values oriented from coast
    var1_rev = da_smooth.load().isel(points_numbers=slice(None, None, -1)).values
    var2 = xr.DataArray(
        var1_rev,
        dims=["x", "time"],
        coords={"x":dists_rev, "time":dates, "lon":("x", lons_track_rev), "lat":("x", lats_track_rev)}
    )
    if units_in == "m":
        var2.coords["x"].attrs["units"] = "m"
    elif units_in == "km":
        var2.coords["x"].attrs["units"] = "km"
    return var2


def apply_optimal_operator(ds, p=-6, q=5, dt=1.0):
    """
    Applies the Powell & Leben (2004) optimal difference operator to an 
    X-TRACK dataset along the 'points_numbers' dimension.
    """
    def get_optimal_weights(p, q, dt):
        """Derives coefficients cn using Lagrange multipliers."""
        p_abs, q_abs = abs(p), abs(q)
        N = p_abs + q_abs
        T = N + 1
        # Relative indices excluding the current point (0)
        n = np.array([i for i in range(-p_abs, q_abs + 1) if i != 0])
        # Build Lagrange Matrix A [cite: 471-478]
        A = np.zeros((T, T))
        A[N, :N] = 1.0  # Constraint: sum(cn) = 1 [cite: 58, 693]
        for i in range(N):
            # c_n derivation from the Lagrange function [cite: 85, 477]
            A[i, :N] = 1.0 / (n * (-n[i] / 2.0))
            A[i, N] = (n[i]**2 * dt**2) / 4.0
            A[i, i] = -1.0
        B = np.zeros(T)
        B[N] = 1.0
        # Solve for coefficients [cite: 88, 483]
        results = np.linalg.solve(A, B)
        return results[:N], n
    # 1. Calculate the weights
    cn, n_indices = get_optimal_weights(p, q, dt)
    # 2. Extract the SLA variable [cite: 1422]
    # Dimensions: (points_numbers: 210, cycles_numbers: 186)
    sla = ds.sla.values 
    points_dim, cycles_dim = sla.shape
    # Initialize output array with NaNs [cite: 1422]
    slopes = np.full(sla.shape, np.nan)
    # 3. Apply operator along the points_numbers dimension
    # Window constraints to avoid index errors at boundaries [cite: 161]
    p_abs = abs(p)
    q_abs = abs(q)
    for i in range(p_abs, points_dim - q_abs):
        # Calculate derivative: sum( (cn / (n*dt)) * (h_{i+n} - h_i) ) [cite: 55, 690]
        # This computes the across-track geostrophic velocity anomaly [cite: 700]
        h_i = sla[i, :]
        h_rel = sla[i + n_indices, :] # Shape: (N, cycles_numbers)
        # Element-wise slope calculation for all cycles at once
        # Slope = sum_{n} [ cn * (h_{i+n} - h_i) / (n * dt) ]
        window_slopes = (h_rel - h_i) / (n_indices[:, np.newaxis] * dt)
        slopes[i, :] = np.sum(cn[:, np.newaxis] * window_slopes, axis=0)
    # Return as a new xarray DataArray to maintain metadata
    return xr.DataArray(
        slopes, 
        coords=ds.sla.coords, 
        dims=ds.sla.dims, 
        name="slope"
    )


p = -6
q = 5
str1 = f"{abs(p)}{abs(q)}"
para = str1

os.makedirs(f"sla_{method}_{para}", exist_ok=True)

sat = "GFO"
pass1 = "008"
f = f"../data/{sat}/ctoh.sla.ref.{sat}.nindian.{pass1}.nc"
ds = xr.open_dataset(f, decode_times=False)
track_number = ds.Pass
print(f"{sat} {track_number} -------------------")
slope = ds.sla.differentiate("points_numbers")
slope_smooth = apply_optimal_operator(ds, p=-6, q=5, dt=1.0)

slope1 = slope.isel(cycles_numbers=0)
slope_smooth1 = slope_smooth.isel(cycles_numbers=0)

slope1.plot(label="slope regular sla")
slope_smooth1.plot(label="odo slope")
plt.legend()
plt.show()
plt.close("all")

sla_asn = track_dist_time_asn_midx(ds, var_str="sla", units_in="m")
slope_asn = sla_asn.differentiate("x")
slope_asn1 = slope_asn.isel(time=0)

slope_smooth_asn = track_dist_time_asn_midx_da(ds, slope_smooth, units_in="m")
slope_smooth_asn1 = slope_asn.isel(time=0)# + 0.05

slope_asn1.plot(label="slope regular sla")
slope_smooth_asn1.plot(label="odo slope")
plt.legend()
plt.show()
plt.close("all")

gc1 = compute_geostrophy_gc(ds, sla_asn)
gc11 = gc1.isel(time=0)

gc2 = compute_geostrophy_gc_from_slope(ds, slope_smooth_asn)
gc21 = gc1.isel(time=0)

gc11.plot(label="regular gc")
gc21.plot(label="odo gc")

plt.legend()
plt.show()
plt.close("all")
