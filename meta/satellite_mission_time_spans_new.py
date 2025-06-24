from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from dateutil.relativedelta import relativedelta

missions = pd.read_csv("data_coverage_df.csv")
print(missions)

colors = ["red", "green", "blue", "purple"]
date_format = "%Y-%m-%d"

min_start_date = min(pd.to_datetime(missions["first_date"]))
max_end_date = max(pd.to_datetime(missions["last_date"]))
min_start_date1 = min_start_date - relativedelta(years=1)
max_end_date1 = max_end_date + relativedelta(years=1)
# print(min_start_date1, max_end_date1)

fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

for index, row in missions.iterrows():
    sat, track_nos, start_date, end_date, tfreq, xfreq = row.loc[
        ["sat", "trackn", "first_date", "last_date", "tfreq", "xfreq"]
    ]
    sln = len(sat)
    words = [sat, str(track_nos), str(tfreq), str(round(xfreq, 1))]
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
    # current_start_time = start_date_o + relativedelta(years=4)
    current_start_time = start_date_o + duration / 4
    for k, word in enumerate(words):
        sln = len(word)
        text_obj = ax.text(
            current_start_time,
            index,
            word,
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=12,
            color=colors[k],
        )
        if k == 0:
            yrs = int(0.21 * sln) + 2
            mns = int(2.53 * sln) + 10
        else:
            yrs = int(0.21 * sln) + 1.5
            mns = int(2.53 * sln) + 8
        # current_start_time = current_start_time + relativedelta(years=yrs)
        current_start_time = current_start_time + relativedelta(months=mns)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
# ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_xlabel("years")
ax.set_xlim(min_start_date1, max_end_date1)
ax.set_title("Satellite Missions Timeline")
ax.invert_yaxis()

satellite = mpatches.Patch(color="red", label="satellite")
tracks = mpatches.Patch(color="green", label="no. of tracks")
tfreq = mpatches.Patch(color="blue", label="temporal frequency")
xfreq = mpatches.Patch(color="purple", label="spatial resolution")
plt.legend(handles=[satellite, tracks, tfreq, xfreq])

plt.savefig(f"satellite_mission_time_spans_new_fig01.png")
# plt.tight_layout()
plt.show()
