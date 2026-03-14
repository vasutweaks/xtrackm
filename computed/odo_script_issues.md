# Issues Found in `c.odo_smooth.one.py`

## 1) Critical: ODO coefficient equation is implemented incorrectly

In `c.odo_smooth.one.py`:

```python
A[i, :N] = 1.0 / (n * (-n[i] / 2.0))
```

But MATLAB/reference form in `odo_filter.m` is proportional to:

```matlab
A(i, :) = [ (1./n) .* (-n(i)/2), (n(i)^2 * dt^2 / 4) ];
```

This is not equivalent (reciprocal vs multiplication), and it changes ODO weights drastically.

## 2) Critical: derivative units are wrong for geostrophic conversion

In `c.odo_smooth.one.py`:

```python
slope_odo = apply_optimal_operator(ds, p=-6, q=5, dt=1.0)
...
gc2 = compute_geostrophy_gc_from_slope(ds, slope_odo_asn)
```

`dt=1.0` here is point index spacing, not physical distance. But later `g/f` multiplies slope as if it were `dη/dx` in SI units. This yields scale errors unless `dt` is in meters (or variable-distance form is implemented).

## 3) High: plotted “ODO gc” is actually regular gc due to typo

In `c.odo_smooth.one.py`:

```python
gc1 = compute_geostrophy_gc(ds, sla_asn)
...
gc2 = compute_geostrophy_gc_from_slope(ds, slope_odo_asn)
gc21 = gc1.isel(time=0)
```

`gc21` should come from `gc2`, not `gc1`.

## 4) High: `p,q` configuration is inconsistent/misleading

In `c.odo_smooth.one.py`:

```python
p = -5
q = 6
para = f"{abs(p)}{abs(q)}"
...
slope_odo = apply_optimal_operator(ds, p=-6, q=5, dt=1.0)
```

Label/parameter says `56`, actual run uses `65`.

## 5) Medium: equatorial mask is one-sided

In `c.odo_smooth.one.py`:

```python
f3 = np.where(lats_track_rev > 2, f2, np.nan)
```

This masks only north of 2°N. For general usage, `abs(lat) > 2` is usually intended.

