import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc
import sympy as sp

base1 = 2

x = sp.symbols('x')
a = sp.symbols('a', constant=True)

f = a**x
fa = f.subs(a, base1)
expon = sp.lambdify(x, fa)

# Find the differential
fa_dx = sp.diff(fa, x)
expon_dx = sp.lambdify(x, fa_dx)

x = np.linspace(0, 5, 500)
y = expon(x)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, y, label="$y=2^x$")

# Point of tangency P
x0 = 2.5
y0 = expon(x0)

# Compute derivative at x0 (slope of tangent)
# m_tangent = np.log(2) * y0  # y' = ln(2) * 2^x
m_tangent = expon_dx(x0)

# Tangent line at P
x_tangent = np.linspace(x0 - 1, x0 + 1, 100)
y_tangent = m_tangent * (x_tangent - x0) + y0
ax.plot(x_tangent, y_tangent, label="Tangent at P", linestyle="--", color="blue")

# Mark point P
ax.plot(x0, y0, "ko")  # Plot point P as black dot
plt.text(x0 + 0.1, y0, " P", ha="left", va="bottom")

# Draw line T1 passing through P making angle 110 degrees with x-axis
theta_deg = 110
theta_rad = np.deg2rad(theta_deg)
m_T1 = np.tan(theta_rad)
x_T1 = np.linspace(x0 - 2, x0 + 2, 100)
y_T1 = m_T1 * (x_T1 - x0) + y0
ax.plot(x_T1, y_T1, label="$T_1$", color="red")

m_T1 = (y_T1[-1] - y_T1[0]) / (x_T1[-1] - x_T1[0])
m_T1_perp = -1 / m_T1
# m_T1_perp = 1.76
x_perp = x_T1
y_perp = m_T1_perp * (x_perp - x0) + y0
# plt.plot(x_perp, y_perp, "r--", linewidth=1)
# t1 = (y_perp[-1] - y_perp[0])/(x_perp[-1] - x_perp[0])
# t2 = (y_T1[-1] - y_T1[0])/(x_T1[-1] - x_T1[0])

ax.annotate(
    "",
    xy=(x_perp[0], y_perp[0]),
    xytext=(x_perp[-1], y_perp[-1]),
    arrowprops=dict(arrowstyle="<->", linestyle="--", color="r"),
)

# Mark the angle theta that T1 makes with the x-axis
arc_radius = 0.5
arc = Arc(
    (x0, y0),
    arc_radius,
    arc_radius,
    angle=0,
    theta1=0,
    theta2=theta_deg,
    edgecolor="black",
)
ax.add_patch(arc)

# Position for theta symbol
theta_label_x = x0 + arc_radius * np.cos(theta_rad / 2)
theta_label_y = y0 + arc_radius * np.sin(theta_rad / 2)
plt.text(theta_label_x, theta_label_y, r"$\theta$", fontsize=14)

# At point P, draw vertical arrow (u bar) pointing both sides
arrow_length = 1
ax.annotate(
    "",
    xy=(x0, y0 + arrow_length),
    xytext=(x0, y0 - arrow_length),
    arrowprops=dict(arrowstyle="<->", color="black"),
)
plt.text(x0 + 0.1, y0 + arrow_length + 0.1, r"$\vec{u}$", fontsize=12)

# At point P, draw horizontal arrow (v bar) pointing both sides
ax.annotate(
    "",
    xy=(x0 - arrow_length, y0),
    xytext=(x0 + arrow_length, y0),
    arrowprops=dict(arrowstyle="<->", color="black"),
)
plt.text(x0 + arrow_length + 0.1, y0 + 0.1, r"$\vec{v}$", fontsize=12)

# plt.xlim(x0 - 2, x0 + 2)
# plt.xlim(1, 4)
ax.set_ylim(y0 - 3, y0 + 3)
ax.set_aspect('equal', adjustable='box')


ax.set_xlabel("")
ax.set_ylabel("")
# plt.legend()
plt.grid(True)
# plt.title("Plot of $y=2^x$ with Tangent at P and Line $T_1$ at $110^\circ$")

plt.savefig("p.schematic.coastal_slope.png", bbox_inches="tight", dpi=300)
plt.show()
