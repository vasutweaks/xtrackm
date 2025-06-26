import matplotlib.pyplot as plt
import numpy as np

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Parameters for the parallel line
y_parallel = 5  # Height above x-axis
x_width = 10  # Width of the parallel line

# Draw the parallel thin line to the x-axis
ax.plot(
    [-x_width / 2, x_width / 2],
    [y_parallel, y_parallel],
    "b-",
    lw=1,
    label="Parallel Line",
)

# Define the traversing line with an angle of 110 degrees
angle = 77  # Angle in degrees (counterclockwise from x-axis)
slope = np.tan(np.radians(angle))  # Slope of the line
x_traverse = np.array([-6, 6])  # X-range of the traversing line
y_traverse = slope * x_traverse  # Y-values for the traversing line

# Plot the traversing line
ax.plot(x_traverse, y_traverse, "r-", lw=1, label="Traversing Line")

# Calculate intersection point P
x_intersect = y_parallel / slope
y_intersect = y_parallel

# Plot and annotate the intersection point P
ax.plot(x_intersect, y_intersect, "ko", markersize=4)  # Mark intersection point
ax.text(x_intersect + 0.2, y_intersect + 0.2, "P", fontsize=14)

# Label the angle θ near the x-axis
ax.text(-0.2, -1.8, r"$\theta$", fontsize=14)

# Draw vertical and horizontal arrows at the intersection point (P)
# Vertical arrow (u-bar)
ax.arrow(
    x_intersect,
    y_intersect,
    0,
    2,
    head_width=0.2,
    head_length=0.4,
    fc="g",
    ec="g",
)
ax.arrow(
    x_intersect,
    y_intersect,
    0,
    -2,
    head_width=0.2,
    head_length=0.4,
    fc="g",
    ec="g",
)
ax.text(
    x_intersect + 0.2,
    y_intersect + 2,
    r"$\bar{v}$",
    fontsize=14,
    color="green",
)
ax.text(
    x_intersect + 0.2,
    y_intersect - 2,
    r"$-\bar{v}$",
    fontsize=14,
    color="green",
)

# Horizontal arrow (v-bar)
ax.arrow(
    x_intersect,
    y_intersect,
    2,
    0,
    head_width=0.2,
    head_length=0.4,
    fc="g",
    ec="g",
)
ax.arrow(
    x_intersect,
    y_intersect,
    -2,
    0,
    head_width=0.2,
    head_length=0.4,
    fc="g",
    ec="g",
)
ax.text(
    x_intersect + 2,
    y_intersect + 0.2,
    r"$\bar{u}$",
    fontsize=14,
    color="green",
)
ax.text(
    x_intersect - 2.4,
    y_intersect + 0.2,
    r"$-\bar{u}$",
    fontsize=14,
    color="green",
)

# Compute the perpendicular slope (negative reciprocal of the traversing line's slope)
perpendicular_slope = -1 / slope

# Define the x-range for the perpendicular line (centered at P)
length = 6  # Length of the dotted perpendicular line in both directions
x_perpendicular = np.array([x_intersect - length / 2, x_intersect + length / 2])

# Compute the corresponding y-values for the perpendicular line
y_perpendicular = perpendicular_slope * (x_perpendicular - x_intersect) + y_intersect

# Draw the thin, dotted perpendicular line with arrows
ax.plot(x_perpendicular, y_perpendicular, 'k--', lw=1)  # Dotted line

# Calculate direction vectors for arrow alignment
dx = (x_perpendicular[1] - x_perpendicular[0]) / 10  # Small increment for arrow length
dy = (y_perpendicular[1] - y_perpendicular[0]) / 10  # Small increment for arrow length

# Add arrows along the perpendicular line pointing correctly in both directions
ax.arrow(x_perpendicular[0], y_perpendicular[0], -dx, -dy, 
         head_width=0.2, head_length=0.2, fc='k', ec='k')

ax.arrow(x_perpendicular[1], y_perpendicular[1], dx, dy, 
         head_width=0.2, head_length=0.2, fc='k', ec='k')

ax.text(
    x_perpendicular[1] + 0.5,
    y_perpendicular[1] + 0.0,
    r"$\bar{g}$",
    fontsize=14,
    color="k",
)
ax.text(
    x_perpendicular[0] - 0.5,
    y_perpendicular[0] - 0.5,
    r"$-\bar{g}$",
    fontsize=14,
    color="k",
)
# Set axis limits and aspect ratio
ax.set_xlim(-6, 7)
ax.set_ylim(-2, 10)
ax.set_aspect('equal', 'box')

# Add grid, legend, and labels
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
plt.savefig('p.schematic.matplotlib.lines.acute.png', bbox_inches='tight', dpi=300)

# Display the plot
plt.show()
