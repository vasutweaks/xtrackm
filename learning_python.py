"""
We can load a shapefile using the cartopy.io.shapereader module instead of 
downloading ne file.
"""
import cartopy.io.shapereader as shpreader
# Load Natural Earth 110m coastline
shpfilename = shpreader.natural_earth(
    resolution='50m',
    category='physical',
    name='coastline'
)

"""What is this"""
merged: pd.DataFrame | None = None
df[[key_col, metric]].rename(columns={metric: label})
merged = merged.merge(frame, on=key_col, how=join_mode)
from itertools import zip_longest
from numbers import Number
if df["track_self"].dtype == object:
    df["track_self"] = df["track_self"].astype(int)
lat_array = np.atleast_1d(latitude)
da = da.expand_dims(transect=[i])

def resolve_guv_file(guv_dir: Path, sat: str, track_self: int, track_other: int, method: str, frac: str) -> Path:
    t1 = str(int(track_self))
    t2 = str(int(track_other))
    candidates = [
        guv_dir / f"guv_at_intersection_{sat}_{t1}_{t2}_{method}_{frac}.nc",
        guv_dir / f"guv_at_intersection_{sat}_{t2}_{t1}_{method}_{frac}.nc",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "No guv file found for tracks "
        f"({t1}, {t2}). Tried: " + ", ".join(str(x) for x in candidates)
    )

