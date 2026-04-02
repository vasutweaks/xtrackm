import math

def estimate_rossby_radius(lat_deg, c1=2.64):
    """
    Roughly estimate the first-baroclinic Rossby radius of deformation.
    Parameters:
    -----------
    lat_deg : float
        Latitude in degrees.
    c1 : float
        Phase speed of the first baroclinic gravity wave in m/s. 
        (Default is 2.64 m/s, typical for the Northern Indian Ocean).
    Returns:
    --------
    rossby_radius_km : float
        Estimated Rossby radius of deformation in kilometers.
    """
    # Constants
    omega = 7.292e-5      # Earth's rotation rate in rad/s
    R_earth = 6378137.0   # Earth's equatorial radius in meters
    # Convert latitude to radians
    lat_rad = math.radians(lat_deg)
    # Calculate Coriolis parameter (f) and its meridional derivative (beta)
    # f = 2 * omega * sin(lat)
    # beta = df/dy = 2 * omega * cos(lat) / R_earth
    f = 2 * omega * math.sin(lat_rad)
    beta = (2 * omega * math.cos(lat_rad)) / R_earth
    # Calculate Rossby radius based on latitude distance from the equator
    if abs(lat_deg) >= 5.0:
        # Mid-to-high latitudes: R = c / |f|
        radius_m = c1 / abs(f)
    else:
        # Equatorial region: R = sqrt(c / beta)
        radius_m = math.sqrt(c1 / beta)
    return radius_m / 1000.0  # Convert meters to kilometers

# --- Examples based on literature values ---
print(f"Equator (0°): {estimate_rossby_radius(0):.1f} km") 
print(f"(15°N): {estimate_rossby_radius(15):.1f} km") 
print(f"Mid-latitudes (45°N): {estimate_rossby_radius(45):.1f} km")
