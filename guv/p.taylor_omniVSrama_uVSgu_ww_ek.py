"""
Plot Taylor diagrams for gu and gu+u_ek vs RAMA/OMNI buoy u-velocity.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geocat.viz as gv

RAMA_EK_CSV = "rama_guv_c_nearest_global_matches.ek.csv"
OMNI_EK_CSV = "omni_guv_c_nearest_global_matches.ek.csv"

OUT_PNG = "taylor_omniVSrama_uVSgu_ww_ek.png"
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


def get_stats(csv_path, y_col, with_ekman=False):
    df = pd.read_csv(csv_path)
    df = filter_by_distance(df)

    gu = df["gu"].to_numpy(dtype=float)
    u_ek = df["u_ek"].to_numpy(dtype=float)
    obs = df[y_col].to_numpy(dtype=float)

    if with_ekman:
        model = gu + u_ek
    else:
        model = gu

    valid = np.isfinite(model) & np.isfinite(obs)
    model = model[valid]
    obs = obs[valid]

    if model.size < 3:
        return np.nan, np.nan, np.nan

    std_model = np.std(model, ddof=0)
    std_obs = np.std(obs, ddof=0)
    corr = np.corrcoef(model, obs)[0, 1]

    return std_model, corr, std_obs


def main():
    # Calculate stats for all 4 cases
    rama_no_std, rama_no_corr, rama_obs_std = get_stats(RAMA_EK_CSV, "rama_u", with_ekman=False)
    rama_ek_std, rama_ek_corr, _ = get_stats(RAMA_EK_CSV, "rama_u", with_ekman=True)

    omni_no_std, omni_no_corr, omni_obs_std = get_stats(OMNI_EK_CSV, "omni_u", with_ekman=False)
    omni_ek_std, omni_ek_corr, _ = get_stats(OMNI_EK_CSV, "omni_u", with_ekman=True)

    # Normalize by observation std
    # We will use one Taylor diagram with normalized std (Ref = 1.0)
    rama_no_std_norm = rama_no_std / rama_obs_std
    rama_ek_std_norm = rama_ek_std / rama_obs_std
    omni_no_std_norm = omni_no_std / omni_obs_std
    omni_ek_std_norm = omni_ek_std / omni_obs_std

    print(f"RAMA: gu         | corr: {rama_no_corr:.3f}, std_norm: {rama_no_std_norm:.3f}")
    print(f"RAMA: gu + u_ek  | corr: {rama_ek_corr:.3f}, std_norm: {rama_ek_std_norm:.3f}")
    print(f"OMNI: gu         | corr: {omni_no_corr:.3f}, std_norm: {omni_no_std_norm:.3f}")
    print(f"OMNI: gu + u_ek  | corr: {omni_ek_corr:.3f}, std_norm: {omni_ek_std_norm:.3f}")

    fig = plt.figure(figsize=(10, 8))
    
    # Normalized Taylor Diagram
    # refstd = 1.0 because we normalized everything
    taylor = gv.TaylorDiagram(fig=fig, label="Reference", refstd=1.0)

    # Add correlation grids
    taylor.add_corr_grid(np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]))
    
    # Add RMS circles
    taylor.add_contours(levels=5, colors="lightgrey")

    # Plot RAMA points
    # gu
    taylor.add_model_set(
        [rama_no_std_norm],
        [rama_no_corr],
        color="royalblue",
        marker="o",
        label="RAMA: gu",
        s=100,
    )
    # gu + u_ek
    taylor.add_model_set(
        [rama_ek_std_norm],
        [rama_ek_corr],
        color="royalblue",
        marker="s",
        label="RAMA: gu + u_ek",
        s=100,
    )

    # Plot OMNI points
    # gu
    taylor.add_model_set(
        [omni_no_std_norm],
        [omni_no_corr],
        color="tomato",
        marker="o",
        label="OMNI: gu",
        s=100,
    )
    # gu + u_ek
    taylor.add_model_set(
        [omni_ek_std_norm],
        [omni_ek_corr],
        color="tomato",
        marker="s",
        label="OMNI: gu + u_ek",
        s=100,
    )

    taylor.add_legend(loc="upper right", fontsize=10)
    
    plt.title("Normalized Taylor Diagram: Altimetry vs Buoys (u-velocity)", y=1.08, fontsize=14)
    
    plt.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
