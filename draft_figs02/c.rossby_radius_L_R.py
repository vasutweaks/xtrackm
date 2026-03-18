import numpy as np

def rossby_radius_nio(latitude, c=2.7):
    """
    Computes the first baroclinic Rossby radius of deformation (in km)
    as a function of latitude for Northern Indian Ocean conditions.
    Parameters:
    -----------
    latitude : float or numpy.ndarray
        Latitude in degrees (Positive for North, e.g., 0 to 30 for NIO).
    c : float
        Phase speed of the first baroclinic mode in m/s.
        Default is 2.7 m/s, typical for the Arabian Sea / Bay of Bengal.
    Returns:
    --------
    L_R_km : float or numpy.ndarray
        Rossby radius of deformation in kilometers.
    """
    # Geophysical constants
    omega = 7.2921e-5     # Earth's angular velocity in rad/s
    R_earth = 6371000.0   # Earth's mean radius in meters
    # Convert latitude to radians and take absolute value 
    # (dynamics are symmetric across the equator)
    lat_rad = np.radians(np.abs(latitude))
    # Coriolis parameter (f)
    f = 2 * omega * np.sin(lat_rad)
    # Equatorial beta parameter (change in Coriolis parameter with latitude at equator)
    beta = 2 * omega / R_earth
    # Calculate Equatorial Rossby Radius (for regions very close to the equator)
    # R_eq = sqrt(c / beta)
    R_eq = np.sqrt(c / beta)
    # Calculate Mid-latitude Rossby Radius
    # To avoid division by zero at exactly 0 degrees, we mask the array
    # If a scalar is passed, we convert it to an array for safe handling
    lat_array = np.atleast_1d(latitude)
    f_array = np.atleast_1d(f)
    L_R = np.zeros_like(lat_array, dtype=float)
    for i in range(len(lat_array)):
        if np.abs(lat_array[i]) < 2.0:
            # Within ~2 degrees of the equator, the standard f-plane approximation 
            # breaks down. We use the equatorial deformation radius instead.
            L_R[i] = R_eq
        else:
            # Standard mid-latitude formula: L_R = c / |f|
            L_R[i] = c / np.abs(f_array[i])
    # Convert from meters to kilometers
    L_R_km = L_R / 1000.0
    # Return scalar if input was scalar
    return L_R_km[0] if np.isscalar(latitude) else L_R_km

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Calculate at a specific latitude (e.g., 15°N, middle of Arabian Sea)
    lat_15 = 15.0
    radius_15 = rossby_radius_nio(lat_15)
    print(f"Rossby Radius at {lat_15}°N: {radius_15:.1f} km")
    # Example 2: Array of latitudes across the Northern Indian Ocean
    lats = np.array([5, 10, 15, 20, 25])
    radii = rossby_radius_nio(lats)
    print("Latitudes (°N):", lats)
    print("Radii (km):    ", np.round(radii, 1))
