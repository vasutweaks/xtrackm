import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Outlier cleaning
# -------------------------------------------------------------------
def clean_outliers(gu, ve, ek=None, max_speed=3.0, mad_thresh=3.5):
    gu = np.asarray(gu)
    ve = np.asarray(ve)

    mask = np.isfinite(gu) & np.isfinite(ve)

    if ek is not None:
        ek = np.asarray(ek)
        mask &= np.isfinite(ek)

    # Physical bounds
    mask &= (np.abs(gu) <= max_speed) & (np.abs(ve) <= max_speed)
    if ek is not None:
        mask &= (np.abs(ek) <= max_speed)

    def mad_outlier(x, thresh):
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad == 0:
            return np.zeros_like(x, dtype=bool)
        z = 0.6745 * (x - med) / mad
        return np.abs(z) > thresh

    mask &= ~mad_outlier(gu, mad_thresh)
    mask &= ~mad_outlier(ve, mad_thresh)
    if ek is not None:
        mask &= ~mad_outlier(ek, mad_thresh)

    if ek is None:
        return gu[mask], ve[mask]
    else:
        return gu[mask], ve[mask], ek[mask]


# -------------------------------------------------------------------
# Plot helper
# -------------------------------------------------------------------
def scatter_with_fit(ax, x, y, xlabel, title):
    ax.scatter(x, y, alpha=0.5)
    ax.grid(True)

    # Correlation
    corr = np.corrcoef(x, y)[0, 1]

    ax.text(
        0.05, 0.95, f"Correlation: {corr:.2f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.5),
    )

    # Trend line
    m, b = np.polyfit(x, y, 1)
    xs = np.sort(x)
    ax.plot(xs, m * xs + b, color="red", lw=2, label="Trend line")

    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
sat = sys.argv[1]

# Read data
df_no_ek = pd.read_csv(
    f"track_guvVSnearby_drifters_at_intersection_points_{sat}_2.csv"
)

df_ek = pd.read_csv(
    f"track_guvVSnearby_drifters_at_intersection_points_{sat}_w_ekman.csv"
)

# Clean data
gu1, ve1 = clean_outliers(df_no_ek["gu"], df_no_ek["ve"])
gu2, ve2, ek2 = clean_outliers(df_ek["gu"], df_ek["ve"], df_ek["ek"])
gu_ek = gu2 + ek2

# -------------------------------------------------------------------
# Figure: side-by-side, shared y-axis
# -------------------------------------------------------------------
fig, axes = plt.subplots(
    ncols=2,
    nrows=1,
    figsize=(14, 6),
    sharey=True
)

# Left: gu vs ve
scatter_with_fit(
    axes[0],
    gu1,
    ve1,
    xlabel="Track gu (m/s)",
    title="gu vs drifter ve"
)
axes[0].set_ylabel("Drifter ve (m/s)")

# Right: gu+ek vs ve
scatter_with_fit(
    axes[1],
    gu_ek,
    ve2,
    xlabel="Track (gu + ek) (m/s)",
    title="gu + ek vs drifter ve"
)

plt.tight_layout()
plt.savefig(f"scatter_{sat}_gu_vs_ve_and_guEk_vs_ve.png", dpi=300)
plt.show()
