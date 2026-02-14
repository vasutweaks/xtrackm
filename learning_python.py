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
