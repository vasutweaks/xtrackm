import matplotlib.pyplot as plt
import numpy as np

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Define angles in degrees and calculate slopes for the two lines
angle1 = 70  # Angle of the first line with the x-axis
angle2 = 110  # Angle of the second line with the x-axis

slope1 = np.tan(np.radians(angle1))  # Slope of the first line
slope2 = np.tan(np.radians(angle2))  # Slope of the second line

# Define starting points for the two lines on the x-axis
x_start1, y_start1 = -2, 0  # Start of Line 1 on the x-axis
x_start2, y_start2 = 2, 0  # Start of Line 2 on the x-axis

# Define the x-range for both lines (extending from their respective starting points)
x_range1 = np.linspace(x_start1, x_start1 + 6, 100)
x_range2 = np.linspace(x_start2, x_start2 - 6, 100)

# Calculate the y-values for both lines using their slopes
y_line1 = slope1 * (x_range1 - x_start1) + y_start1
y_line2 = slope2 * (x_range2 - x_start2) + y_start2

# Plot the two lines
(line1, ) = ax.plot(x_range1, y_line1, "r-", lw=1, label=f"Line 1 (acute)")
(line2, ) = ax.plot(x_range2, y_line2, "b-", lw=1, label=f"Line 2 (obtuse)")

# Find the intersection point of the two lines
# Solve y_line1 = y_line2 to find the intersection (numerically)
A = np.array([[slope1, -1], [slope2, -1]])
b = np.array([slope1 * x_start1, slope2 * x_start2])
x_intersect, y_intersect = np.linalg.solve(A, b)

# Mark and label the intersection point P
ax.plot(x_intersect, y_intersect, "ko",
        markersize=4)  # Mark intersection point
ax.text(x_intersect + 0.2, y_intersect + 0.2, "P", fontsize=14)

# Draw vertical and horizontal arrows at the intersection point P
# Vertical arrow (v-bar)
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
    x_intersect - 0.1,
    y_intersect + 2.5,
    r"$\bar{v}$",
    fontsize=14,
    color="green",
)

# Horizontal arrow (u-bar)
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
    x_intersect + 2.5,
    y_intersect - 0.1,
    r"$\bar{u}$",
    fontsize=14,
    color="green",
)

# Draw perpendicular dotted arrows to each line at the intersection point P
# Perpendicular slope for Line 1 (negative reciprocal of slope1)
perp_slope1 = -1 / slope1
x_perp1 = np.linspace(x_intersect - 3, x_intersect + 3, 100)
y_perp1 = perp_slope1 * (x_perp1 - x_intersect) + y_intersect

# Plot the perpendicular line to Line 1 with the same color (Red)
ax.plot(x_perp1, y_perp1, "r--",
        lw=1)  # Dotted line for perpendicular to Line 1

dx = (x_perp1[1] - x_perp1[0]) / 10  # Small increment for arrow length
dy = (y_perp1[1] - y_perp1[0]) / 10  # Small increment for arrow length

# Add arrows along the perpendicular line pointing correctly in both directions
ax.arrow(x_perp1[0], y_perp1[0], -dx, -dy, 
         head_width=0.2, head_length=0.2, fc='r', ec='r')

ax.arrow(x_perp1[-1], y_perp1[-1], dx, dy, 
         head_width=0.2, head_length=0.2, fc='r', ec='r')
ax.text(
    x_perp1[-1] + 0.5,
    y_perp1[-1] + 0.0,
    r"$\bar{g}$",
    fontsize=14,
    color="r",
)
ax.text(
    x_perp1[0] - 0.5,
    y_perp1[0] - 0.5,
    r"$-\bar{g}$",
    fontsize=14,
    color="r",
)
# Perpendicular slope for Line 2 (negative reciprocal of slope2)
perp_slope2 = -1 / slope2
x_perp2 = np.linspace(x_intersect - 3, x_intersect + 3, 100)
y_perp2 = perp_slope2 * (x_perp2 - x_intersect) + y_intersect

# Plot the perpendicular line to Line 2 with the same color (Blue)
ax.plot(x_perp2, y_perp2, "b--",
        lw=1)  # Dotted line for perpendicular to Line 2

dx = (x_perp2[1] - x_perp2[0]) / 10  # Small increment for arrow length
dy = (y_perp2[1] - y_perp2[0]) / 10  # Small increment for arrow length

# Add arrows along the perpendicular line pointing correctly in both directions
ax.arrow(x_perp2[0], y_perp2[0], -dx, -dy, 
         head_width=0.2, head_length=0.2, fc='b', ec='b')

ax.arrow(x_perp2[-1], y_perp2[-1], dx, dy, 
         head_width=0.2, head_length=0.2, fc='b', ec='b')

ax.text(
    x_perp2[-1] + 0.5,
    y_perp2[-1] + 0.0,
    r"$\bar{g}$",
    fontsize=14,
    color="b",
)
ax.text(
    x_perp2[0] - 0.5,
    y_perp2[0] - 0.5,
    r"$-\bar{g}$",
    fontsize=14,
    color="b",
)
# Set axis limits and aspect ratio
ax.set_xlim(-6, 6)
ax.set_ylim(0, 12)
ax.set_aspect("equal", "box")

# Add grid, legend, and labels
ax.grid(True, linestyle="--", alpha=0.7)
ax.legend(loc="upper right")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")

plt.savefig("p.schematic.matplotlib_lines_g1_g2.png", bbox_inches="tight")
# Display the plot
plt.show()
