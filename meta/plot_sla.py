
import os
import netCDF4
import matplotlib.pyplot as plt
import numpy as np

def draw_sla_plots():
    """
    Draws 2D filled plots of Sea Level Anomaly (SLA) from NetCDF files.
    """
    input_dir = '../data/TPN+J1N+J2N+J3N'
    output_dir = '../plots/TPN_J1N_J2N_J3N_sla_plots'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.nc'):
            try:
                filepath = os.path.join(input_dir, filename)
                with netCDF4.Dataset(filepath, 'r') as nc_file:
                    lat = nc_file.variables['lat'][:]
                    lon = nc_file.variables['lon'][:]
                    sla = nc_file.variables['sla'][:]
                    # Calculate the mean of sla across the cycles dimension
                    sla = np.ma.mean(sla, axis=1)

                    # Create a 2D filled plot
                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(lon, lat, c=sla, cmap='jet', vmin=-0.5, vmax=0.5)
                    plt.colorbar(scatter, label='SLA (m)')
                    plt.xlabel('Longitude')
                    plt.ylabel('Latitude')
                    plt.title(f'SLA for {filename}')
                    
                    # Save the plot
                    output_filename = os.path.splitext(filename)[0] + '.png'
                    output_path = os.path.join(output_dir, output_filename)
                    plt.savefig(output_path)
                    plt.close()
                    
                    print(f"Plot saved to {output_path}")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == '__main__':
    draw_sla_plots()
