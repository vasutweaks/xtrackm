import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# codex resume 019cd8ab-eee9-72e0-9952-b9e4a8bf1b11

RAMA_NO_EK_CSV = "rama_guv_c_nearest_global_matches.csv"
RAMA_EK_CSV = "rama_guv_c_nearest_global_matches.ek.csv"
OMNI_NO_EK_CSV = "omni_guv_c_nearest_global_matches.csv"
OMNI_EK_CSV = "omni_guv_c_nearest_global_matches.ek.csv"

OUT_PNG = "scatter_omniVSrama_uVSgu_ww_ek.png"
MIN_DISTANCE_KM = None
MAX_DISTANCE_KM = 250.0


def filter_by_distance(df):
    dist = df["distance_km"].to_numpy(dtype=float)
    keep = np.isfinite(dist)
    if MIN_DISTANCE_KM is not None:
        keep &= dist >= float(MIN_DISTANCE_KM)
    if MAX_DISTANCE_KM is not None:
        keep &= dist <= float(MAX_DISTANCE_KM)
    return df.loc[keep].copy()


def load_xy(csv_path, y_col, with_ekman=False):
    df = pd.read_csv(csv_path)
    df = filter_by_distance(df)

    x = df["gu"].to_numpy(dtype=float)
    if with_ekman:
        x = x + df["u_ek"].to_numpy(dtype=float)

    y = df[y_col].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    return x, y, len(df)


def print_variance_decomposition(csv_path, label, total_col):
    df = filter_by_distance(pd.read_csv(csv_path))
    gu = df["gu"].to_numpy(dtype=float)
    u_ek = df["u_ek"].to_numpy(dtype=float)
    total_ref = df[total_col].to_numpy(dtype=float)
    valid = np.isfinite(gu) & np.isfinite(u_ek) & np.isfinite(total_ref)
    gu = gu[valid]
    u_ek = u_ek[valid]
    total_ref = total_ref[valid]

    if gu.size < 2:
        print(f"{label} variance decomposition: insufficient points")
        return

    total = gu + u_ek
    var_total = np.var(total, ddof=0)
    var_total_ref = np.var(total_ref, ddof=0)
    var_gu = np.var(gu, ddof=0)
    var_uek = np.var(u_ek, ddof=0)
    cov_gu_uek = np.mean((gu - np.mean(gu)) * (u_ek - np.mean(u_ek)))
    rhs = var_gu + var_uek + 2.0 * cov_gu_uek

    if var_total_ref == 0.0:
        n_var_gu = np.nan
        n_var_uek = np.nan
        n_2cov = np.nan
    else:
        n_var_gu = var_gu / var_total_ref
        n_var_uek = var_uek / var_total_ref
        n_2cov = (2.0 * cov_gu_uek) / var_total_ref

    print(f"\n{label} variance decomposition (N={gu.size})")
    print(f"Var({total_col}) [total ref]   = {var_total_ref:.8f}")
    print(f"Var(gu + u_ek) [LHS]      = {var_total:.8f}")
    print(f"Var(gu)                    = {var_gu:.8f}")
    print(f"Var(u_ek)                  = {var_uek:.8f}")
    print(f"Cov(gu, u_ek)              = {cov_gu_uek:.8f}")
    print(f"Var(gu)+Var(u_ek)+2*Cov    = {rhs:.8f}")
    print(f"Closure error (LHS - RHS)  = {var_total - rhs:.8e}")
    print(f"Normalized by Var({total_col}):")
    print(f"  Var(gu)/Var({total_col})   = {n_var_gu:.6f}")
    print(f"  Var(u_ek)/Var({total_col}) = {n_var_uek:.6f}")
    print(f"  2*Cov/Var({total_col})     = {n_2cov:.6f}")
    print(f"  Sum normalized terms        = {n_var_gu + n_var_uek + n_2cov:.6f}")


def rmse(x, y):
    if x.size == 0 or y.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((y - x) ** 2)))


def print_rmse_change(label, x_before, y_before, x_after, y_after):
    rmse_before = rmse(x_before, y_before)
    rmse_after = rmse(x_after, y_after)
    delta = rmse_after - rmse_before
    pct = np.nan
    if np.isfinite(rmse_before) and rmse_before != 0.0:
        pct = 100.0 * delta / rmse_before

    print(f"\n{label} RMSE change (y vs x)")
    print(f"RMSE before Ekman = {rmse_before:.4f} m/s")
    print(f"RMSE after Ekman  = {rmse_after:.4f} m/s")
    print(f"Change (after-before) = {delta:+.4f} m/s ({pct:+.2f}%)")


def add_panel(ax, x, y, title, ylabel, xlabel, extent=None):
    if x.size < 3:
        ax.text(
            0.5,
            0.98,
            f"{title} (insufficient points)",
            transform=ax.transAxes,
            va="top",
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.5, edgecolor="white"),
            color="white",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        return None

    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    r = np.corrcoef(x, y)[0, 1]
    stderr = np.sqrt(np.sum((y - yhat) ** 2) / (x.size - 2))

    hb = ax.hexbin(x, y, gridsize=55, cmap="viridis", mincnt=1, extent=extent)

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
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.55, edgecolor="white"),
        color="white",
    )

    ax.text(
        0.6,
        0.98,
        title,
        transform=ax.transAxes,
        va="top",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.5, edgecolor="white"),
        color="white",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    return hb


def main():
    min_lbl = MIN_DISTANCE_KM if MIN_DISTANCE_KM is not None else "-inf"
    max_lbl = MAX_DISTANCE_KM if MAX_DISTANCE_KM is not None else "+inf"
    x_label_no_ek = "Altimetry gu (m/s)"
    x_label_ek = "Altimetry gu + u_ek (m/s)"
    none = ""

    x_rama_no, y_rama_no, n_rama_no = load_xy(RAMA_NO_EK_CSV, "rama_u", with_ekman=False)
    x_rama_ek, y_rama_ek, n_rama_ek = load_xy(RAMA_EK_CSV, "rama_u", with_ekman=True)
    x_omni_no, y_omni_no, n_omni_no = load_xy(OMNI_NO_EK_CSV, "omni_u", with_ekman=False)
    x_omni_ek, y_omni_ek, n_omni_ek = load_xy(OMNI_EK_CSV, "omni_u", with_ekman=True)
    all_x = np.concatenate([x_rama_no, x_rama_ek, x_omni_no, x_omni_ek])
    all_y = np.concatenate([y_rama_no, y_rama_ek, y_omni_no, y_omni_ek])
    global_extent = (
        float(np.nanmin(all_x)),
        float(np.nanmax(all_x)),
        float(np.nanmin(all_y)),
        float(np.nanmax(all_y)),
    )

    fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True, constrained_layout=True)

    hb00 = add_panel(
        axs[0, 0],
        x_rama_no,
        y_rama_no,
        f"RAMA u vs gu (no Ekman) | distance_km in [{min_lbl}, {max_lbl}]",
        "RAMA u (m/s)",
        none,
        extent=global_extent,
    )
    hb01 = add_panel(
        axs[0, 1],
        x_rama_ek,
        y_rama_ek,
        f"RAMA u vs gu+u_ek (with Ekman) | distance_km in [{min_lbl}, {max_lbl}]",
        none,
        none,
        extent=global_extent,
    )
    hb10 = add_panel(
        axs[1, 0],
        x_omni_no,
        y_omni_no,
        f"OMNI u vs gu (no Ekman) | distance_km in [{min_lbl}, {max_lbl}]",
        "OMNI u (m/s)",
        x_label_no_ek,
        extent=global_extent,
    )
    hb11 = add_panel(
        axs[1, 1],
        x_omni_ek,
        y_omni_ek,
        f"OMNI u vs gu+u_ek (with Ekman) | distance_km in [{min_lbl}, {max_lbl}]",
        none,
        x_label_ek,
        extent=global_extent,
    )

    top_hbs = [hb for hb in [hb00, hb01] if hb is not None]
    if top_hbs:
        top_vmax = max(float(np.nanmax(hb.get_array())) for hb in top_hbs)
        for hb in top_hbs:
            hb.set_clim(1.0, top_vmax)
        cbar_top = fig.colorbar(top_hbs[0], ax=axs[0, :], location="right", fraction=0.03, pad=0.02)
        cbar_top.set_label("Count")

    bottom_hbs = [hb for hb in [hb10, hb11] if hb is not None]
    if bottom_hbs:
        bottom_vmax = max(float(np.nanmax(hb.get_array())) for hb in bottom_hbs)
        for hb in bottom_hbs:
            hb.set_clim(1.0, bottom_vmax)
        cbar_bottom = fig.colorbar(bottom_hbs[0], ax=axs[1, :], location="right", fraction=0.03, pad=0.02)
        cbar_bottom.set_label("Count")

    fig.savefig(OUT_PNG, dpi=180)
    plt.show()
    plt.close(fig)

    print(f"Rows after distance filter | RAMA no-ek: {n_rama_no}, RAMA ek: {n_rama_ek}")
    print(f"Rows after distance filter | OMNI no-ek: {n_omni_no}, OMNI ek: {n_omni_ek}")
    print_rmse_change("RAMA", x_rama_no, y_rama_no, x_rama_ek, y_rama_ek)
    print_rmse_change("OMNI", x_omni_no, y_omni_no, x_omni_ek, y_omni_ek)
    print_variance_decomposition(RAMA_EK_CSV, "RAMA (with Ekman file)", "rama_u")
    print_variance_decomposition(OMNI_EK_CSV, "OMNI (with Ekman file)", "omni_u")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
