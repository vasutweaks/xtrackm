import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = "radar_valid_ru_gu_OR.csv"
OUT_PNG = "scatter_radar_u_vs_gu_ek.png"
MIN_DISTANCE_KM = None
MAX_DISTANCE_KM = 250.0


df = pd.read_csv(CSV_PATH)

x1 = df["gu"].to_numpy(dtype=float)
x2 = df["ek_u"].to_numpy(dtype=float)
x = x1 + x2
y = df["ru"].to_numpy(dtype=float)
valid = np.isfinite(x) & np.isfinite(y)
x = x[valid]
y = y[valid]

if x.size < 3:
    print(
        f"Only {x.size} valid points after distance filter "
        f"(min={MIN_DISTANCE_KM}, max={MAX_DISTANCE_KM})."
    )
else:
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    r = np.corrcoef(x, y)[0, 1]
    stderr = np.sqrt(np.sum((y - yhat) ** 2) / (x.size - 2))

    fig, ax = plt.subplots(figsize=(9, 7))
    hb = ax.hexbin(x, y, gridsize=55, cmap="viridis", mincnt=1)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Count")

    xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    ax.plot(xs, slope * xs + intercept, color="crimson", linewidth=2.0, label="Linear fit")
    ax.plot(xs, xs, color="white", linestyle="--", linewidth=1.2, label="1:1 line")

    text = (
        f"N = {x.size}\n"
        f"r = {r:.3f}\n"
        f"Std error = {stderr:.3f} m/s\n"
        f"y = {slope:.3f}x + {intercept:.3f}"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.55, edgecolor="white"),
        color="white",
    )

    ax.set_title(f"")
    ax.set_xlabel("Altimetry gu (m/s)")
    ax.set_ylabel("radar u (m/s)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    plt.show()
    plt.close(fig)

    print(f"Rows after distance filter: {len(df)}")
    print(f"Saved {OUT_PNG}")
