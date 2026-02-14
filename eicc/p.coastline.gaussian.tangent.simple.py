"""Simple standalone example for coastline_gaussian_tangent().

Inputs:
- query_point
- sigma (Gaussian smoothing)
- optional region_bbox_tuple = (lon_min, lat_min, lon_max, lat_max)

Outputs:
- heading angle in degrees
- tangent_vector
- nearest_point
- distance
"""

from tools_shapely import coastline_gaussian_tangent


# Required input: (lon, lat)
query_point = (83.62, 17.17)

# Optional input: Gaussian smoothing sigma
sigma = 3

# Optional input: clipping bbox tuple
# Default used by the function is (70, 5, 95, 30)
region_bbox_tuple = (70, 5, 95, 30)


result = coastline_gaussian_tangent(
    query_point=query_point,
    sigma=sigma,
    region_bbox_tuple=region_bbox_tuple,
)

print("heading_angle_deg:", result["heading_angle_deg"])
print("tangent_vector:", result["tangent_vector"])
print("nearest_point:", result["nearest_point"])
print("distance:", result["distance"])
