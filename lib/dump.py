
def compute_geostrophy_v(ds, sla_smooth, current="meridional"):
    lons_track_rev = ds.lon.values[::-1]
    lats_track_rev = ds.lat.values[::-1]
    if len(lons_track_rev) == 0:
        return
    # Compute the angle using arctan2 for correct sign and quadrant
    delta_lat = lats_track_rev[-1] - lats_track_rev[0]
    delta_lon = lons_track_rev[-1] - lons_track_rev[0]
    angle_r = np.arctan2(delta_lat, delta_lon)  # Angle in radians

    # Compute Coriolis parameter and gravity
    f = cor_p(lats_track_rev).values
    f = np.where(np.abs(lats_track_rev) > 2, f, np.nan)  # Mask equatorial band
    g = grav(lats_track_rev).values
    ratio = (g / f)[:, np.newaxis]

    # Gradient of SLA along the track (s-direction)
    dhbyds = sla_smooth.differentiate("x")  # "x" is the distance along the track

    # Geostrophic current magnitude along the track
    gc = ratio * dhbyds

    # Compute zonal and meridional components using correct trigonometry
    if current == "zonal":
        u = -gc * np.sin(angle_r)
        return u
    elif current == "meridional":
        # v = gc * np.cos(angle_r)
        u = -gc * np.sin(angle_r)
        v = (gc + u*np.sin(angle_r))/np.cos(angle_r)
        return v
    elif current == "speed":
        u = gc * np.sin(angle_r)
        v = -gc * np.cos(angle_r)
        speed = np.sqrt(u**2 + v**2)
        return speed


def compute_geostrophy_atan2(ds, sla_smooth, current="zonal"):
    lons_track = ds.lon.values
    lats_track = ds.lat.values
    lons_track_rev = ds.lon.values[::-1]
    lats_track_rev = ds.lat.values[::-1]
    lon_coast = lons_track[-1]  # this on coast
    lat_coast = lats_track[-1]  # this on coast
    lon_equator = lons_track[0]  # this on equator
    lat_equator = lats_track[0]  # this on equator
    if len(lons_track) == 0:
        return
    angle_r2 = np.arctan2(lat_coast - lat_equator, lon_coast - lon_equator)
    angle2 = np.rad2deg(angle_r2)
    f1 = cor_p(lats_track_rev)
    f1 = f1.where(lats_track_rev > 2, np.nan)
    # 2s-2n band equatorial band is masked according to Sudre 2013
    g1 = grav(lats_track_rev)  # ms-1
    ratio = g1 / f1
    ratio1 = ratio.values
    ratio2 = ratio1[:, np.newaxis]
    dhbydx = sla_smooth.differentiate("x")
    gc = -1 * ratio2 * dhbydx
    if current == "zonal":
        gu = -1 * gc * np.sin(angle_r2)
        return gu
    elif current == "meridional":
        gv = gc * np.cos(angle_r2)
        return gv
    elif current == "speed":
        gu = -1 * gc * np.sin(angle_r2)
        gv = gc * np.cos(angle_r2)
        gs = (gu**2 + gv**2) ** 0.5
        return gs
    elif current == "gc":
        gc = ratio2 * dhbydx
        return gc


def compute_geostrophy(ds, sla_smooth, current="zonal"):
    lons_track_rev = ds.lon.values[::-1]
    lats_track_rev = ds.lat.values[::-1]
    if len(lons_track_rev) == 0:
        return
    m = (lats_track_rev[-1] - lats_track_rev[0]) / (
        lons_track_rev[-1] - lons_track_rev[0]
    )
    angle_r = np.arctan(m)
    angle = np.rad2deg(angle_r)
    f1 = cor_p(lats_track_rev)
    f2 = f1.values
    f3 = np.where(lats_track_rev > 2, f2, np.nan)
    # 2s-2n band equatorial band is masked according to Sudre 2013
    g = grav(lats_track_rev)  # ms-1
    g1 = g.values
    ratio = g1 / f3
    ratio1 = ratio[:, np.newaxis]
    dhbydx = sla_smooth.differentiate("x")
    gc = -1 * ratio1 * dhbydx
    if current == "zonal":
        gu = -1 * gc * np.cos(np.deg2rad(90.0 - abs(angle)))
        return gu
    elif current == "meridional":
        gu = gc * np.sin(np.deg2rad(90.0 - abs(angle)))
        return gu
    elif current == "speed":
        guu = -1 * gc * np.cos(np.deg2rad(90.0 - abs(angle)))
        guv = gc * np.sin(np.deg2rad(90.0 - abs(angle)))
        gu = (guu**2 + guv**2) ** 0.5
        return gu


def geostrophic_current_components_slopes(g1, g2, m1, m2):
    """
    Calculates the zonal (u) and meridional (v) components and the speed of
    the geostrophic current.
    Parameters:
    - g1: Cross-track geostrophic current along track T1 (float)
    - g2: Cross-track geostrophic current along track T2 (float)
    - m1: Slope of track T1 (float)
    - m2: Slope of track T2 (float)
    Returns:
    - u: Zonal component of the geostrophic current (float)
    - v: Meridional component of the geostrophic current (float)
    - speed: Speed (magnitude) of the geostrophic current (float)
    """
    # Convert slopes to angles in radians
    theta1 = math.atan(m1)
    theta2 = math.atan(m2)
    print(theta1, theta2)
    sin_theta1 = math.sin(theta1)
    cos_theta1 = math.cos(theta1)
    sin_theta2 = math.sin(theta2)
    cos_theta2 = math.cos(theta2)
    sin_theta_diff = math.sin(theta2 - theta1)
    if sin_theta_diff == 0:
        raise ValueError("The tracks are parallel or the angle difference is \
                         zero, cannot compute components.")
    u = (cos_theta2 * g1 - cos_theta1 * g2.interp(time=g1.time)) / sin_theta_diff
    v = (sin_theta2 * g1 - sin_theta1 * g2.interp(time=g1.time)) / sin_theta_diff
    speed = (u**2 + v**2)**0.5
    u1 = u
    v1 = v
    speed1 = speed
    return u1, v1, speed1


def geostrophic_components_from_a(gc1, gc2, a1, a2):
    """
    Compute zonal and meridional geostrophic current components from
    cross-track velocities and track slopes of two intersecting satellite tracks.
    Parameters:
    -----------
    gc1 : float or array-like
        Cross-track geostrophic velocity from track 1 (m/s)
    gc2 : float or array-like
        Cross-track geostrophic velocity from track 2 (m/s)
    slope1 : float
        Slope of track 1 (dy/dx = rise/run)
    slope2 : float
        Slope of track 2 (dy/dx = rise/run)
    Returns:
    --------
    u : float or array-like
        Zonal (eastward) velocity component (m/s)
    v : float or array-like
        Meridional (northward) velocity component (m/s)
    Raises:
    -------
    ValueError: If tracks are parallel (same slope)
    """
    # Calculate intersection angle
    theta = a2 - a1
    sin_theta = math.sin(theta)
    # Additional check using sine of intersection angle
    if abs(sin_theta) < 1e-6:
        raise ValueError(
            f"Tracks are nearly parallel (intersection angle = {math.degrees(theta):.2f}°). "
            "Cannot resolve both velocity components.")
    # Calculate trigonometric values
    cos_a1 = math.cos(a1)
    cos_a2 = math.cos(a2)
    sin_a1 = math.sin(a1)
    sin_a2 = math.sin(a2)
    # Solve for velocity components using the derived formulas:
    # u = (gc1 * cos(a2) - gc2 * cos(a1)) / sin(a2 - a1)
    # v = (gc2 * sin(a1) - gc1 * sin(a2)) / sin(a2 - a1)
    u = (gc1 * cos_a2 - gc2 * cos_a1) / sin_theta
    v = (gc2 * sin_a1 - gc1 * sin_a2) / sin_theta
    u = -1 * u
    return u, v


