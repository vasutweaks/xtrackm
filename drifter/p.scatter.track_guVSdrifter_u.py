import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sat = "TP+J1+J2+J3+S6A"
df = pd.read_csv(f"track_guvVSnearby_drifters_at_intersection_points_{sat}.csv")

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

# Add the trend line to the plot
plt.plot(df['gu'], m*df['gu'] + b, color='red', linewidth=2, label='Trend Line')

plt.legend()

# Save the plot
plt.savefig(f'scatter_plot_{sat}.png')
plt.show()

print(f"Correlation coefficient: {correlation}")
