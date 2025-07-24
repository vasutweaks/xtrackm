import pickle
from tools_xtrackm import *
import rich
from rich.console import Console

console = Console()

ln=len(coastal_d.keys())
total_tuples = []
for id1 in coastal_d.keys():
    lon_coastal, lat_coastal = coastal_d[id1]
    if not is_within_region(lon_coastal, lat_coastal, *TRACKS_REG):
        continue
    for sat in sats_new[:]:
        list_tuples = coastal_nearby_track_box(id1, sat, box_size=1.5)
        print(list_tuples)
        if len(list_tuples) == 1 and all(
            element is None for element in list_tuples[0]
        ):
            continue
        console.print(f"{id1} {sat} ---------------------")
        console.print(list_tuples)
        total_tuples.extend(list_tuples)

sorted_list = sorted(total_tuples, key=lambda x: x[4])
print(sorted_list)

with open("closest_tracks_all_coastal.pkl", "wb") as file:
    pickle.dump(sorted_list, file)
