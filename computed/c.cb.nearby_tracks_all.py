import pickle
from tools_xtrackm import *
import rich
from rich.console import Console

console = Console()

total_tuples = []

ids = list(cb_d.keys())
ln = len(ids)

for i in range(ln):
    id1 = ids[i]
    lon_cb, lat_cb = cb_d[id1]
    # if not is_within_region(lon_cb, lat_cb, *TRACKS_REG):
    #     continue
    for sat in sats_new[:]:
        list_tuples = cb_nearby_track_box(id1, sat, box_size=1.5)
        # print(list_tuples[0])
        if len(list_tuples) == 1 and all(
            element is None for element in list_tuples[0]
        ):
            continue
        console.print(f"{id1} {sat} ---------------------")
        console.print(list_tuples)
        total_tuples.extend(list_tuples)

sorted_list = sorted(total_tuples, key=lambda x: x[4])
print(sorted_list)

with open("closest_tracks_all_cb_new.pkl", "wb") as file:
    pickle.dump(sorted_list, file)
