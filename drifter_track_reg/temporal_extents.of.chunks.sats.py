from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from namesm import sats_new


# all buoys' time range is within following ranges
chunk_time_dict = {
    "1_5000": (
        np.datetime64("1985-10-30T00:00:00.000000000"),
        np.datetime64("2001-09-26T00:00:00.000000000"),
    ),
    "5001_10000": (
        np.datetime64("1997-03-20T00:00:00.000000000"),
        np.datetime64("2009-04-25T00:00:00.000000000"),
    ),
    "10001_15000": (
        np.datetime64("2006-03-05T00:00:00.000000000"),
        np.datetime64("2012-11-04T00:00:00.000000000"),
    ),
    "15001_current": (
        np.datetime64("2010-09-30T00:00:00.000000000"),
        np.datetime64("2025-05-19T00:00:00.000000000"),
    ),
}

DATE_FORMAT = "%Y-%m-%d"


def load_missions(csv_path="data_coverage_df.csv"):
    return pd.read_csv(csv_path)


def _get_satellite_extent(missions, sat_name):
    sat_rows = missions[missions["sat"] == sat_name]
    if sat_rows.empty:
        available = ", ".join(sorted(missions["sat"].unique()))
        raise ValueError(f"Satellite '{sat_name}' not found. Available: {available}")

    row = sat_rows.iloc[0]
    start_date = datetime.strptime(row["first_date"], DATE_FORMAT)
    end_date = datetime.strptime(row["last_date"], DATE_FORMAT)
    return start_date, end_date


def plot_all_temporal_extents(
    missions,
    chunk_ranges,
    output_path="temporal_extents_chunks_sats.png",
    show=True,
):
    mission_starts = pd.to_datetime(missions["first_date"])
    mission_ends = pd.to_datetime(missions["last_date"])
    chunk_starts = pd.to_datetime([v[0] for v in chunk_ranges.values()])
    chunk_ends = pd.to_datetime([v[1] for v in chunk_ranges.values()])

    min_start_date = min(mission_starts.min(), chunk_starts.min())
    max_end_date = max(mission_ends.max(), chunk_ends.max())
    min_start_date1 = min_start_date - relativedelta(years=1)
    max_end_date1 = max_end_date + relativedelta(years=1)

    fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")

    y_positions = []
    y_labels = []

    # Plot satellites first
    for sat_idx, row in missions.iterrows():
        sat = row["sat"]
        start_date = datetime.strptime(row["first_date"], DATE_FORMAT)
        end_date = datetime.strptime(row["last_date"], DATE_FORMAT)
        duration = end_date - start_date

        ax.barh(
            sat_idx,
            duration,
            left=start_date,
            color="skyblue",
            edgecolor="black",
            alpha=0.9,
        )
        ax.text(
            start_date + duration / 2,
            sat_idx,
            sat,
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=10,
            color="black",
        )
        y_positions.append(sat_idx)
        y_labels.append(f"SAT: {sat}")

    # Plot chunks on new rows below satellites
    chunk_offset = len(missions)
    for chunk_idx, (chunk, (start_date64, end_date64)) in enumerate(chunk_ranges.items()):
        y = chunk_offset + chunk_idx
        start_date = pd.to_datetime(start_date64).to_pydatetime()
        end_date = pd.to_datetime(end_date64).to_pydatetime()
        duration = end_date - start_date

        ax.barh(
            y,
            duration,
            left=start_date,
            color="salmon",
            edgecolor="black",
            alpha=0.9,
        )
        ax.text(
            start_date + duration / 2,
            y,
            chunk,
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=10,
            color="black",
        )
        y_positions.append(y)
        y_labels.append(f"CHUNK: {chunk}")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Year")
    ax.set_xlim(min_start_date1, max_end_date1)
    ax.set_title("Temporal Extents of Satellite Missions and Chunks")
    ax.invert_yaxis()

    satellite_patch = mpatches.Patch(color="skyblue", label="Satellite")
    chunk_patch = mpatches.Patch(color="salmon", label="Chunk")
    ax.legend(handles=[satellite_patch, chunk_patch])

    plt.savefig(output_path, dpi=200)
    if show:
        plt.show()


def plot_single_chunk_sat_extent(
    chunk_name,
    sat_name,
    missions,
    chunk_ranges,
    output_path="temporal_extent_single_chunk_single_sat.png",
    show=True,
):
    if chunk_name not in chunk_ranges:
        available = ", ".join(chunk_ranges.keys())
        raise ValueError(f"Chunk '{chunk_name}' not found. Available: {available}")

    sat_start, sat_end = _get_satellite_extent(missions, sat_name)
    chunk_start64, chunk_end64 = chunk_ranges[chunk_name]
    chunk_start = pd.to_datetime(chunk_start64).to_pydatetime()
    chunk_end = pd.to_datetime(chunk_end64).to_pydatetime()

    min_start = min(sat_start, chunk_start) - relativedelta(years=1)
    max_end = max(sat_end, chunk_end) + relativedelta(years=1)

    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")

    sat_duration = sat_end - sat_start
    ax.barh(
        0,
        sat_duration,
        left=sat_start,
        color="skyblue",
        edgecolor="black",
        alpha=0.9,
    )
    ax.text(
        sat_start + sat_duration / 2,
        0,
        sat_name,
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=10,
        color="black",
    )

    chunk_duration = chunk_end - chunk_start
    ax.barh(
        1,
        chunk_duration,
        left=chunk_start,
        color="salmon",
        edgecolor="black",
        alpha=0.9,
    )
    ax.text(
        chunk_start + chunk_duration / 2,
        1,
        chunk_name,
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=10,
        color="black",
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"SAT: {sat_name}", f"CHUNK: {chunk_name}"])
    ax.set_xlabel("Year")
    ax.set_xlim(min_start, max_end)
    ax.set_title(f"Temporal Extents: {sat_name} and {chunk_name}")
    ax.invert_yaxis()

    satellite_patch = mpatches.Patch(color="skyblue", label="Satellite")
    chunk_patch = mpatches.Patch(color="salmon", label="Chunk")
    ax.legend(handles=[satellite_patch, chunk_patch])

    plt.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def plot_pairwise_chunk_sat_extents(
    missions,
    chunk_ranges,
    sat_names=None,
    output_dir="pairwise_temporal_extents",
    show=False,
):
    if sat_names is None:
        sat_names = list(missions["sat"].unique())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for chunk_name in chunk_ranges.keys():
        for sat_name in sat_names:
            plot_single_chunk_sat_extent(
                chunk_name,
                sat_name,
                missions,
                chunk_ranges,
                output_path=str(output_path / f"{chunk_name}__{sat_name}.png"),
                show=show,
            )


if __name__ == "__main__":
    missions_df = load_missions()
    plot_pairwise_chunk_sat_extents(
        missions_df,
        chunk_time_dict,
        sat_names=sats_new,
        output_dir="pairwise_temporal_extents",
        show=True,
    )
