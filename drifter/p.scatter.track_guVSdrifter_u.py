import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

sat_here = sys.argv[1]
df = pd.read_csv(f"track_guvVSnearby_drifters_at_intersection_points_{sat_here}.csv")

# Drop rows with missing values in 'gu' or 've' columns
df.dropna(subset=["gu", "ve"], inplace=True)

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df["gu"], df["ve"], alpha=0.5)
plt.title("Scatter Plot of gu vs ve")
plt.xlabel("track gu")
plt.ylabel("drifter ve")
plt.grid(True)

# Calculate the correlation coefficient
correlation = df["gu"].corr(df["ve"])

# Place the correlation coefficient value on the plot
plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

# Calculate the trend line (line of best fit)
m, b = np.polyfit(df['gu'], df['ve'], 1)
import statsmodels.api as sm
X = sm.add_constant(df['gu'])
model = sm.RLM(df['ve'], X, M=sm.robust.norms.HuberT()).fit()
b, m = model.params
x_sorted = np.sort(df['gu'])


# Add the trend line to the plot
# plt.plot(x_sorted, m*x_sorted + b, color='red', linewidth=2, label='Trend Line')
plt.plot([df['gu'].min(), df['gu'].max()],
         [m * df['gu'].min() + b, m * df['gu'].max() + b],
         color='red', linewidth=2, label='Trend Line')


plt.legend()

# Save the plot
plt.savefig(f'scatter_plot_{sat_here}_track_uVSnearby_drifter_u.png')
plt.show()

print(f"Correlation coefficient: {correlation}")
