import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from tools_xtrackm import *
from geopy import distance

_VALID_METHODS = ("gaussian", "savgol")

# copilot --resume=1108ce5c-68e7-40f0-95f7-b8e8708e8e56

def isobath_heading_at_point(
    topo,
    level,
    query_point,
    method="gaussian",
    sigma=5.0,
    window_length=21,
    polyorder=3,
    step=10,
    lon_name="lon",
    lat_name="lat",
):
    """
    Compute the isobath tangent heading angle at a query point.
    Combines three steps in one call:
      1. Extract all (lon, lat) points along the requested isobath using
         matplotlib contour tracing.
      2. Smooth the raw polyline with the chosen filter and subsample every
         *step* points.
      3. Find the nearest smoothed point to *query_point* and return the local
         tangent direction as an angle measured counter-clockwise from the
         positive x-axis (east), in degrees, range [-180, 180].
    Smoothing methods
    -----------------
    ``'gaussian'``  (default)
        1-D Gaussian filter via ``scipy.ndimage.gaussian_filter1d``.
        Controlled by *sigma* (std dev in grid points).  Better frequency
        response than a box filter; reflect-padded edges.
    ``'savgol'``
        Savitzky-Golay filter via ``scipy.signal.savgol_filter``.
        Controlled by *window_length* (must be odd) and *polyorder*.
        Fits a local polynomial, so it preserves shape features better
        than Gaussian while also allowing analytical derivative extraction
        (``deriv=1``), making the tangent estimate more accurate.
    Parameters
    ----------
    topo : xarray.DataArray
        2-D bathymetry / topography field with lon and lat coordinates.
    level : float
        Isobath depth level (e.g. -200 for the 200 m isobath).
    query_point : tuple[float, float]
        (lon, lat) of the location of interest.
    method : {'gaussian', 'savgol'}, optional
        Smoothing method (default ``'gaussian'``).
    sigma : float, optional
        Std dev of the Gaussian kernel in grid points (default 5.0).
        Used only when ``method='gaussian'``.
    window_length : int, optional
        Length of the Savitzky-Golay filter window; must be odd and
        greater than *polyorder* (default 21).
        Used only when ``method='savgol'``.
    polyorder : int, optional
        Polynomial order for the Savitzky-Golay fit (default 3).
        Used only when ``method='savgol'``.
    step : int, optional
        Keep every *step*-th point after smoothing (default 10).
    lon_name : str, optional
        Name of the longitude coordinate in *topo* (default ``"lon"``).
    lat_name : str, optional
        Name of the latitude coordinate in *topo* (default ``"lat"``).
    Returns
    -------
    dict with keys:
        heading_deg : float
            Heading angle in degrees, measured counter-clockwise from east
            (positive x-axis).  Range [-180, 180].
        isobath_pts : list[tuple[float, float]]
            Smoothed and subsampled (lon, lat) isobath points, ready for
            plotting.
        nearest_point : tuple[float, float]
            The smoothed isobath point closest to *query_point*.
        nearest_index : int
            Index of *nearest_point* within *isobath_pts*.
        method : str
            The smoothing method used.
    Raises
    ------
    ValueError
        If *method* is not recognised, no isobath points are found, or
        the smoothed polyline has fewer than two points.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"method={method!r} is not recognised. "
            f"Choose one of {_VALID_METHODS}."
        )
    # ------------------------------------------------------------------ #
    # Step 1 – extract raw isobath points via matplotlib contour          #
    # ------------------------------------------------------------------ #
    lon = topo[lon_name].values
    lat = topo[lat_name].values
    z = topo.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    fig, ax = plt.subplots()
    cs = ax.contour(lon2d, lat2d, z, levels=[level])
    plt.close(fig)

    raw_pts = []
    segments = cs.allsegs[0] if cs.allsegs else []
    for seg in segments:
        raw_pts.extend(map(tuple, seg))

    if not raw_pts:
        raise ValueError(
            f"No isobath points found for level={level}. "
            "Check that the level exists within the topo domain."
        )

    arr = np.asarray(raw_pts, dtype=float)

    # ------------------------------------------------------------------ #
    # Step 2 – smooth and subsample                                       #
    # ------------------------------------------------------------------ #
    if method == "gaussian":
        lon_smooth = gaussian_filter1d(arr[:, 0], sigma=sigma)
        lat_smooth = gaussian_filter1d(arr[:, 1], sigma=sigma)
        # Tangent via finite differences on the smoothed positions.
        arr_smooth = np.column_stack((lon_smooth, lat_smooth))
        pts = arr_smooth[::step]
        dlon_arr = None  # computed on demand in step 3
        dlat_arr = None

    else:  # savgol
        # Ensure window_length is odd and feasible.
        wl = window_length if window_length % 2 == 1 else window_length + 1
        wl = max(wl, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)
        wl = min(wl, len(arr))
        if wl % 2 == 0:
            wl -= 1

        lon_smooth = savgol_filter(arr[:, 0], window_length=wl, polyorder=polyorder)
        lat_smooth = savgol_filter(arr[:, 1], window_length=wl, polyorder=polyorder)
        # Analytical first derivative directly from the polynomial fit.
        dlon_smooth = savgol_filter(
            arr[:, 0], window_length=wl, polyorder=polyorder, deriv=1
        )
        dlat_smooth = savgol_filter(
            arr[:, 1], window_length=wl, polyorder=polyorder, deriv=1
        )
        arr_smooth = np.column_stack((lon_smooth, lat_smooth))
        pts = arr_smooth[::step]
        dlon_arr = dlon_smooth[::step]
        dlat_arr = dlat_smooth[::step]

    if len(pts) < 2:
        raise ValueError(
            "Fewer than two points remain after smoothing/subsampling; "
            "reduce smoothing parameters or *step*."
        )

    # ------------------------------------------------------------------ #
    # Step 3 – nearest-point lookup and tangent angle                     #
    # ------------------------------------------------------------------ #
    # q = np.asarray(query_point, dtype=float)
    # d2 = np.sum((pts - q) ** 2, axis=1)
    d2 = [distance.distance((query_point[1], query_point[0]), (pt[1], pt[0])).km ** 2 for pt in pts]
    idx = int(np.argmin(d2))

    if dlon_arr is not None:
        # savgol: use the analytical polynomial derivative directly —
        # more accurate than a finite-difference approximation.
        dx = float(dlon_arr[idx])
        dy = float(dlat_arr[idx])
    else:
        # gaussian: finite differences on the smoothed positions.
        if idx == 0:
            v = pts[1] - pts[0]
        elif idx == len(pts) - 1:
            v = pts[-1] - pts[-2]
        else:
            v = pts[idx + 1] - pts[idx - 1]
        dx, dy = float(v[0]), float(v[1])

    heading_deg = float(np.degrees(np.arctan2(dy, dx)))

    return {
        "heading_deg": heading_deg,
        "isobath_pts": [tuple(xy) for xy in pts],
        "nearest_point": (float(pts[idx, 0]), float(pts[idx, 1])),
        "nearest_index": idx,
        "method": method,
    }


# ====================================================================== #
# Main: slope angle at all coastal points, all satellites                #
# ====================================================================== #
if __name__ == "__main__":
    TRACKS_REG = (43.0, 99, 0, 25.4)
    cmap1 = "twilight_shifted"
    level = -1000
    # Half-length of the tangent segment drawn at each coastal point (degrees)
    SEG_LEN = 0.8

    dse = xr.open_dataset("/home/srinivasu/allData/topo/etopo_60s_io.nc")
    topo = dse.rose.sel(lon=slice(*TRACKS_REG[:2])).sel(lat=slice(*TRACKS_REG[2:]))

    common_kw = dict(topo=topo, level=level, step=15)
    smooth_kw = dict(method="savgol", window_length=41, polyorder=3)

    for sat in sats_new:
        tracks_n = get_total_tracks(sat)
        tsta, tend = get_time_limits(sat)
        fig, ax1 = plt.subplots(
            1, 1,
            subplot_kw={"projection": ccrs.PlateCarree()},
            figsize=(14, 9),
            layout="constrained",
        )
        topo.plot(ax=ax1, add_colorbar=False, add_labels=False, cmap=cmap1)
        decorate_axis(ax1, "", *TRACKS_REG)
        ax1.grid()
        info = f"MISSION: {sat} | {tsta} – {tend} | {tracks_n} tracks | isobath {level} m"
        ax1.text(0.15, 0.95, info, transform=ax1.transAxes,
                 fontsize=11, fontweight="bold")

        coast_lons, coast_lats, headings, track_numbers = [], [], [], []
        iso_plotted = False

        for f in sorted(
            glob.glob(f"/home/srinivasu/xtrackm/data/{sat}/ctoh.sla.ref.{sat}.nindian.*.nc")
        ):
            ds = xr.open_dataset(f, decode_times=False)
            lons_track = ds.lon.values
            lats_track = ds.lat.values
            if len(lons_track) < 2:
                continue

            lon_coast = lons_track[-1]   # coastal end of track
            lat_coast = lats_track[-1]
            track_number = int(ds.Pass)

            if not is_within_region(lon_coast, lat_coast, *NIO):
                continue

            try:
                # query_point is (lon, lat) — matching internal pts ordering
                result = isobath_heading_at_point(
                    query_point=(lon_coast, lat_coast),
                    **common_kw,
                    **smooth_kw,
                )
            except Exception as e:
                print(f"  skip track {track_number}: {e}")
                continue

            heading = result["heading_deg"]
            coast_lons.append(lon_coast)
            coast_lats.append(lat_coast)
            headings.append(heading)
            track_numbers.append(track_number)

            # Draw tangent segment at the coastal point oriented to the isobath
            ang_rad = np.radians(heading)
            dlon = SEG_LEN * np.cos(ang_rad)
            dlat = SEG_LEN * np.sin(ang_rad)
            ax1.plot(
                [lon_coast - dlon, lon_coast + dlon],
                [lat_coast - dlat, lat_coast + dlat],
                color="red", linewidth=1.5, zorder=5,
                transform=ccrs.PlateCarree(),
            )

            # Plot isobath once (same contour for every track in this satellite)
            if not iso_plotted:
                iso_lons, iso_lats = zip(*result["isobath_pts"])
                ax1.scatter(iso_lons, iso_lats, color="cyan", s=3, zorder=3,
                            label=f"{level} m isobath")
                iso_plotted = True

        # Save heading angles to CSV
        if coast_lons:
            df_out = pd.DataFrame({
                "sat": sat,
                "track_number": track_numbers,
                "lon_coast": coast_lons,
                "lat_coast": coast_lats,
                "heading_deg": headings,
            })
            csv_path = f"slopes_coastal_{sat}.csv"
            df_out.to_csv(csv_path, index=False)
            print(f"Saved {len(df_out)} rows → {csv_path}")

        # Scatter coastal points coloured by isobath heading angle
        if coast_lons:
            sc = ax1.scatter(
                coast_lons, coast_lats,
                c=headings, cmap="hsv", vmin=-180, vmax=180,
                s=40, zorder=6, transform=ccrs.PlateCarree(),
                label="coastal point",
            )
            plt.colorbar(sc, ax=ax1,
                         label="Isobath heading at coastal point (°, CCW from east)",
                         shrink=0.6)

        ax1.legend(loc="lower right", fontsize=8)
        fig.savefig(f"slopes_coastal_{sat}.png", dpi=150, bbox_inches="tight")
        # plt.show()
        plt.close("all")
