from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.relativedelta import relativedelta
from namesm import sats_new
from pathlib import Path
import sys
script = Path(sys.argv[0])
png_file = script.with_suffix(".png")
# codex resume 019d70fe-d37c-77c3-bdff-8f2a524c80e7
def plot_mission_timeline(ax):
    missions = pd.read_csv("data_coverage_df.csv")
    colors = ["red", "green", "blue", "purple", "darkorange"]
    date_format = "%Y-%m-%d"

    min_start_date = min(pd.to_datetime(missions["first_date"]))
    max_end_date = max(pd.to_datetime(missions["last_date"]))
    min_start_date1 = min_start_date - relativedelta(years=1)
    max_end_date1 = max_end_date + relativedelta(years=1)

    for index, row in missions.iterrows():
        sat, track_nos, n_intersections, start_date, end_date, tfreq, xfreq = row.loc[
            ["sat", "trackn", "n_intersections", "first_date", "last_date", "tfreq", "xfreq"]
        ]
        words = [sat, str(track_nos), str(tfreq), str(round(xfreq, 1)), str(int(n_intersections))]
        start_date_o = datetime.strptime(start_date, date_format)
        end_date_o = datetime.strptime(end_date, date_format)
        duration = end_date_o - start_date_o
        ax.barh(
            index,
            duration,
            left=start_date_o,
            color="skyblue",
            edgecolor="black",
        )

        current_start_time = start_date_o + duration / 4 - duration * 0.1
        for k, word in enumerate(words):
            ax.text(
                current_start_time,
                index,
                word,
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=12,
                color=colors[k],
            )
            sln = len(word)
            if k == 0:
                mns = int(2.53 * sln) + 10
            else:
                mns = int(2.53 * sln) + 8
            current_start_time = current_start_time + relativedelta(months=mns)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlabel("years")
    ax.set_xlim(min_start_date1, max_end_date1)
    ax.invert_yaxis()
    ax.text(0.01, 0.97, "a)", transform=ax.transAxes, va="top", ha="left", fontsize=13, fontweight="bold")

    satellite = mpatches.Patch(color="red", label="satellite")
    tracks = mpatches.Patch(color="green", label="no. of tracks")
    tfreq = mpatches.Patch(color="blue", label="temporal frequency")
    xfreq = mpatches.Patch(color="purple", label="spatial resolution")
    n_inter = mpatches.Patch(color="darkorange", label="no. of intersections")
    ax.legend(handles=[satellite, tracks, tfreq, xfreq, n_inter], loc="upper right")


def plot_valid_perc(bottom_spec):
    subgs = bottom_spec.subgridspec(1, 2, width_ratios=[4.2, 1.8], wspace=0.0)
    ax = plt.subplot(subgs[0])
    ax_bar = plt.subplot(subgs[1], sharey=ax)

    df_all = pd.read_csv("spatio_temporal_valid_perc.csv")
    cmap = plt.get_cmap("tab10")
    bar_labels = []
    bar_values = []
    bar_colors = []
    legend_handles = []
    legend_labels = []

    for i, sat_name in enumerate(sats_new):
        df = df_all[df_all["sat"] == sat_name].sort_values("track_len")
        df = df.sort_values("perc_valid_xt")
        if df.empty:
            continue

        x = range(len(df))
        y = df["perc_valid_xt"]
        color = cmap(i % cmap.N)
        mean_perc = y.mean()

        line, = ax.plot(x, y, label=sat_name, color=color, linewidth=2.8)
        bar_labels.append(sat_name)
        bar_values.append(mean_perc)
        bar_colors.append(color)
        legend_handles.append(line)
        legend_labels.append(sat_name)

        middle_idx = len(df) // 2
        if sat_name == "S3A":
            middle_idx = len(df) // 3
        ax.text(
            middle_idx,
            y.iloc[middle_idx],
            sat_name,
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="bottom",
            color=color,
        )

    ax.set_xlabel("track number")
    ax.set_ylabel("mean valid points (%)")
    ax.grid(True, color="lightgray", linewidth=0.7, alpha=0.7)
    ax.text(0.01, 0.97, "b)", transform=ax.transAxes, va="top", ha="left", fontsize=13, fontweight="bold")

    bar_data = sorted(zip(bar_values, bar_labels, bar_colors), key=lambda t: t[0])
    sorted_values = [v for v, _, _ in bar_data]
    sorted_labels = [l for _, l, _ in bar_data]
    sorted_colors = [c for _, _, c in bar_data]

    ax_bar.bar(sorted_labels, sorted_values, color=sorted_colors)
    ax_bar.set_xlabel("satellite")
    ax_bar.tick_params(axis="y", labelleft=False)
    ax_bar.set_xticks([])
    ax_bar.grid(True, color="lightgray", linewidth=0.7, alpha=0.7)
    ax_bar.text(0.01, 0.97, "c)", transform=ax_bar.transAxes, va="top", ha="left", fontsize=13, fontweight="bold")

    ax.legend(legend_handles, legend_labels, loc="lower right", fontsize=9, ncol=2, frameon=False)


def main():
    fig = plt.figure(figsize=(13, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.14)

    ax_top = fig.add_subplot(gs[0])
    plot_mission_timeline(ax_top)
    plot_valid_perc(gs[1])

    # out = "combined_mission_and_spatio_temporal_valid_perc.png"
    # plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(f"{png_file}", dpi=300, bbox_inches="tight")
    print(f"Saved {png_file}")
    plt.show()


if __name__ == "__main__":
    main()
