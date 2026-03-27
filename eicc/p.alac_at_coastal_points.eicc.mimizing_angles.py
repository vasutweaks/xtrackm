import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Patch
from tools_shapely import *
from tools_xtrackm import *

# copilot --resume=d419a1e5-878f-477e-813b-3b06f3d9a50b


def solve_theta_cartesian(gu, gv, dim='time', decode_degrees=True):
    """
    Calculates the angle theta (CCW from East/X-axis) that minimizes 
    the perpendicular variance S(theta) = sum((v*cos(theta) - u*sin(theta))**2).
    This identifies the orientation of the Major Axis (Principal Axis).
    Parameters:
    -----------
    u : xr.DataArray
        Zonal velocity component (East-West).
    v : xr.DataArray
        Meridional velocity component (North-South).
    dim : str
        The dimension to sum over (e.g., 'time').
    decode_degrees : bool
        If True, returns the result in degrees. Otherwise, radians.
    Returns:
    --------
    theta : float or xr.DataArray
        The calculated angle.
    """
    # It is standard practice to remove the mean if calculating 
    # the principal axis of fluctuations (variance ellipse)
    u_prime = gu - gu.mean(dim=dim)
    v_prime = gv - gv.mean(dim=dim)
    # Calculate terms for the tan(2*theta) formula
    sum_uv = (u_prime * v_prime).sum(dim=dim)
    sum_u2 = (u_prime**2).sum(dim=dim)
    sum_v2 = (v_prime**2).sum(dim=dim)
    # Using np.arctan2(y, x) handles the quadrant correctly and
    # prevents division by zero if sum_u2 == sum_v2.
    # atan2(2 * sum_uv, sum_u2 - sum_v2)
    two_theta = np.arctan2(2 * sum_uv, sum_u2 - sum_v2)
    theta = 0.5 * two_theta
    if decode_degrees:
        return np.degrees(theta)
    return theta


REG = (62, 92, 0, 30)
dse = xr.open_dataset("/home/srinivasu/allData/topo/etopo_60s_io.nc")
topo = dse.rose.sel(lon=slice(*REG[:2])).sel(lat=slice(*REG[2:]))


def along_across_shore_components(gu, gv, coast_heading_deg):
    """
    Resolve vector (gu, gv) into along-shore and across-shore components.
    Sign convention:
    - coast_heading_deg is measured CCW from +x (East).
    - along_shore is positive in the heading direction.
    - across_shore is positive 90 deg CCW from heading (left-hand side of coast).
      If you need right-hand positive, negate across_shore.
    """
    theta = np.deg2rad(coast_heading_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    # Elementwise projection keeps scalar or array/xarray shape.
    along_shore = gu * c + gv * s
    across_shore = -gu * s + gv * c  # +90 deg CCW
    return along_shore, across_shore


method = "loess"
frac1 = "0.05"

sat_val = "TP+J1+J2+J3+S6A"
sat_val = "S3A"
sat_val = "ERS1+ERS2+ENV+SRL"

df = pd.read_csv(f"intersections_near_east_coast_{sat_val}.csv")

# Skip the first point (lowest latitude)
df_out = df.sort_values("lats_inter").iloc[1:].reset_index(drop=True)

N_PTS = 7
assert len(df_out) == N_PTS, (
    f"Expected {N_PTS} points after skipping first, got {len(df_out)}"
)

POINT_COLORS = [plt.cm.tab10(i) for i in range(N_PTS)]
POINT_LABELS = list("ABCDEFG")

# Map extent: 10N–22N, 78E–88E
MAP_EXTENT = (78.0, 88.0, 10.0, 22.0)

# --- Pre-collect all timeseries data ---
point_data = []
sigma = 3
for i, row in df_out.iterrows():
    sat1 = row["sat"]
    tno_s = str(row["track_self"])
    tno_o = str(row["track_other"])
    lon1 = float(row["lons_inter"])
    lat1 = float(row["lats_inter"])
    query_point = (lon1, lat1)

    f_guv = (
        f"../guv/guv_at_intersections_{method}/"
        f"guv_at_intersection_{sat1}_{tno_s}_{tno_o}_{method}_{frac1}.nc"
    )
    ds_guv = xr.open_dataset(f_guv)
    gu = ds_guv.gu
    gv = ds_guv.gv
    theta = solve_theta_cartesian(gu, gv, dim="time", decode_degrees=True)
    gs = np.sqrt(gu**2 + gv**2)
    print(f"coast_heading_deg={theta:.1f}° at point {i} ({lon1:.2f}E, {lat1:.2f}N)")
    al, ac = along_across_shore_components(gu, gv, theta)
    times = pd.to_datetime(gs.time.values)
    # Monthly climatology (mean of each calendar month across all years)
    gs_clim = gs.groupby("time.month").mean("time")
    gu_clim = gu.groupby("time.month").mean("time")
    al_clim = al.groupby("time.month").mean("time")

    point_data.append(
        dict(
            sat1=sat1,
            tno_s=tno_s,
            tno_o=tno_o,
            lon1=lon1,
            lat1=lat1,
            al=al,
            al_clim=al_clim,
        )
    )

# --- Build figure: left = tall map, right = 7 timeseries rows ---
fig = plt.figure(figsize=(18, 14))
gs_layout = gridspec.GridSpec(
    N_PTS,
    2,
    figure=fig,
    width_ratios=[1, 2.5],
    hspace=0.0,
    wspace=0.3,
)

# --- Left: vertically elongated cartopy map spanning all rows ---
ax_map = fig.add_subplot(gs_layout[:, 0], projection=ccrs.PlateCarree())
ax_map.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
ax_map.add_feature(cfeature.LAND, facecolor="0.93", zorder=0)
ax_map.add_feature(cfeature.OCEAN, facecolor="aliceblue", zorder=0)
ax_map.coastlines(resolution="50m", linewidth=1.0, zorder=2)
gl = ax_map.gridlines(
    draw_labels=True,
    linewidth=0.5,
    color="gray",
    alpha=0.6,
    linestyle="--",
)
gl.top_labels = False
gl.right_labels = False
# ax_map.set_title(f"{sat_val}\nIntersection points", fontsize=9)

# Connect all 7 points with a dashed gray line
lons_pts = [e["lon1"] for e in point_data]
lats_pts = [e["lat1"] for e in point_data]
ax_map.plot(
    lons_pts,
    lats_pts,
    color="gray",
    linewidth=0.8,
    linestyle="--",
    transform=ccrs.PlateCarree(),
    zorder=3,
)

# Plot each point with its assigned color and letter label
for idx, entry in enumerate(point_data):
    color = POINT_COLORS[idx]
    label = POINT_LABELS[idx]
    ax_map.plot(
        entry["lon1"],
        entry["lat1"],
        marker="o",
        markersize=9,
        color=color,
        markeredgecolor="black",
        markeredgewidth=0.8,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    ax_map.text(
        entry["lon1"] + 0.18,
        entry["lat1"] + 0.1,
        label,
        fontsize=10,
        fontweight="bold",
        color=color,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )

# --- Right: 7 monthly climatology subplots, A at bottom (row 6) → G at top (row 0) ---
MONTHS = np.arange(1, 13)
MONTH_NAMES = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

for idx, entry in enumerate(point_data):
    row = N_PTS - 1 - idx  # A → row 6, G → row 0
    ax_ts = fig.add_subplot(gs_layout[row, 1])
    color = POINT_COLORS[idx]
    label = POINT_LABELS[idx]
    al_clim = entry["al_clim"]

    # Seasonal background shading (month-based x-axis)
    ax_ts.axvspan(2 - 0.5, 5 + 0.5, color="yellow", alpha=0.15, zorder=0)   # Feb–May
    ax_ts.axvspan(6 - 0.5, 9 + 0.5, color="cyan",   alpha=0.15, zorder=0)   # Jun–Sep
    ax_ts.axvspan(10 - 0.5, 12 + 0.5, color="gray", alpha=0.15, zorder=0)   # Oct–Dec

    # Monthly climatology bar plot
    # ax_ts.bar(
    #     MONTHS,
    #     gs_clim.values,
    #     color=color,
    #     alpha=0.8,
    #     edgecolor="k",
    #     linewidth=0.5,
    #     zorder=2,
    # )

    ax_ts.plot(
        MONTHS,
        al_clim.values,
        color=color,
        linewidth=3,
        zorder=2,
    )

    # Point label inside left edge of plot
    ax_ts.text(
        0.005, 0.5, label,
        transform=ax_ts.transAxes, fontsize=12,
        fontweight="bold", color=color,
        va="center", ha="left",
    )

    ax_ts.set_xlim(0.5, 12.5)
    all_vals = gs_clim.values[np.isfinite(gs_clim.values)]
    ypad = (all_vals.max() - all_vals.min()) * 0.08
    # ax_ts.set_ylim(-all_vals.max() - ypad, all_vals.max() + ypad)
    # ax_ts.set_ylim(-all_vals.max() - ypad, all_vals.max() + ypad)
    ax_ts.set_ylabel("m/s", fontsize=14)
    ax_ts.set_xticks(MONTHS)
    ax_ts.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
    ax_ts.tick_params(labelsize=12)
    # Only bottom subplot (A, idx=0) shows month labels
    if idx == 0:
        ax_ts.set_xticklabels(MONTH_NAMES)
    else:
        ax_ts.set_xticklabels([])

    # Legend only on the top panel (G, idx=N_PTS-1)
    if idx == N_PTS - 1:
        handles = [
            Patch(facecolor="yellow", alpha=0.4, edgecolor="none", label="Feb–May"),
            Patch(facecolor="cyan",   alpha=0.4, edgecolor="none", label="Jun–Sep"),
            Patch(facecolor="gray",   alpha=0.4, edgecolor="none", label="Oct–Dec"),
        ]
        ax_ts.legend(handles=handles, loc="upper right", fontsize=12, ncol=3)
        ax_ts.set_title("Monthly climatology of |u| (gs)", fontsize=11)

# fig.suptitle(
#     f"Current speed at EICC coastal intersection points — {sat_val}",
#     fontsize=13,
#     fontweight="bold",
#     y=1.002,
# )
out_png = f"speed_timeseries_allinone_{sat_val}.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.show()
plt.close("all")
print(f"Saved: {out_png}")
