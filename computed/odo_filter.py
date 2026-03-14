import numpy as np
import xarray as xr

def apply_optimal_operator(ds, p, q, dt=1.0):
    """
    Applies the Powell & Leben (2004) optimal difference operator to an 
    X-TRACK dataset along the 'points_numbers' dimension.
    """
    
    def get_optimal_weights(p, q, dt):
        """Derives coefficients cn using Lagrange multipliers."""
        p_abs, q_abs = abs(p), abs(q)
        N = p_abs + q_abs
        T = N + 1
        
        # Relative indices excluding the current point (0)
        n = np.array([i for i in range(-p_abs, q_abs + 1) if i != 0])
        
        # Build Lagrange Matrix A [cite: 471-478]
        A = np.zeros((T, T))
        A[N, :N] = 1.0  # Constraint: sum(cn) = 1 [cite: 58, 693]
        
        for i in range(N):
            # c_n derivation from the Lagrange function [cite: 85, 477]
            A[i, :N] = (1.0 / n) * (-n[i] / 2.0)
            A[i, N] = (n[i]**2 * dt**2) / 4.0
            A[i, i] = -1.0
            
        B = np.zeros(T)
        B[N] = 1.0
        
        # Solve for coefficients [cite: 88, 483]
        results = np.linalg.solve(A, B)
        return results[:N], n

    # 1. Calculate the weights
    cn, n_indices = get_optimal_weights(p, q, dt)
    
    # 2. Extract the SLA variable [cite: 1422]
    # Dimensions: (points_numbers: 210, cycles_numbers: 186)
    sla = ds.sla.values 
    points_dim, cycles_dim = sla.shape
    
    # Initialize output array with NaNs [cite: 1422]
    slopes = np.full(sla.shape, np.nan)
    
    # 3. Apply operator along the points_numbers dimension
    # Window constraints to avoid index errors at boundaries [cite: 161]
    p_abs = abs(p)
    q_abs = abs(q)
    
    for i in range(p_abs, points_dim - q_abs):
        # Calculate derivative: sum( (cn / (n*dt)) * (h_{i+n} - h_i) ) [cite: 55, 690]
        # This computes the across-track geostrophic velocity anomaly [cite: 700]
        h_i = sla[i, :]
        h_rel = sla[i + n_indices, :] # Shape: (N, cycles_numbers)
        
        # Element-wise slope calculation for all cycles at once
        # Slope = sum_{n} [ cn * (h_{i+n} - h_i) / (n * dt) ]
        window_slopes = (h_rel - h_i) / (n_indices[:, np.newaxis] * dt)
        slopes[i, :] = np.sum(cn[:, np.newaxis] * window_slopes, axis=0)
        
    # Return as a new xarray DataArray to maintain metadata
    return xr.DataArray(
        slopes, 
        coords=ds.sla.coords, 
        dims=ds.sla.dims, 
        name="geostrophic_velocity_anomaly"
    )

# Example Usage:
# ds = xr.open_dataset("your_xtrack_file.nc")
# # A 12-point filter (p=-5, q=6) is common for WFS mesoscale 
# v_geo = apply_optimal_operator(ds, p=-5, q=6)
