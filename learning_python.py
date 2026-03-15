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
