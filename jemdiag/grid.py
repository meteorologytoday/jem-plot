"""Helpers for identifying grid dimensions and coordinates on jem output variables.

Different jem components describe their grids differently:

- ``atm`` uses dims ``(lon, lat, level)`` with real coordinate values.
- ``ocn`` uses dims ``(longitude, latitude, depth)`` with no coordinate
  values (treated as plain integer indices for now — see
  :mod:`jemdiag.data_loader`).
- ``fakelnd`` uses dims ``(longitude, latitude)`` plus 2D curvilinear
  coordinates ``longitude2D``/``latitude2D``.

These helpers let the plotting and selection code work with any of the
above without hardcoding variable- or component-specific dimension names.
"""

import numpy as np

HORIZONTAL_DIM_PAIRS = [("lon", "lat"), ("longitude", "latitude")]
VERTICAL_DIMS = ["level", "depth"]


def horizontal_dims(da):
    """Return the ``(x_dim, y_dim)`` dimension names for a DataArray."""
    for x_dim, y_dim in HORIZONTAL_DIM_PAIRS:
        if x_dim in da.dims and y_dim in da.dims:
            return x_dim, y_dim
    raise ValueError(
        f"Could not find horizontal (lon/lat) dimensions among {da.dims}"
    )


def vertical_dim(da):
    """Return the vertical dimension name (``level``/``depth``), or ``None``."""
    for dim in VERTICAL_DIMS:
        if dim in da.dims:
            return dim
    return None


def horizontal_coords(da, x_dim, y_dim):
    """Return ``(x, y, is_2d)`` coordinate arrays suitable for ``pcolormesh``.

    If 2D curvilinear coordinates (e.g. ``longitude2D``/``latitude2D``) are
    present, they are used (transposed to ``(y_dim, x_dim)`` order).
    Otherwise the 1D dimension coordinates are returned (these may be real
    coordinate values or integer indices, depending on the component).
    """
    lon2d_name = None
    lat2d_name = None
    for name, coord in da.coords.items():
        if set(coord.dims) == {x_dim, y_dim}:
            lname = str(name).lower()
            if "lon" in lname:
                lon2d_name = name
            elif "lat" in lname:
                lat2d_name = name
    if lon2d_name is not None and lat2d_name is not None:
        lon2d = da.coords[lon2d_name].transpose(y_dim, x_dim)
        lat2d = da.coords[lat2d_name].transpose(y_dim, x_dim)
        return np.asarray(lon2d.values), np.asarray(lat2d.values), True
    return np.asarray(da[x_dim].values), np.asarray(da[y_dim].values), False


def ocean_depth_from_dzt(dzt):
    """Return per-layer physical ocean depth (m, negative below the surface).

    ``dzt`` is layer thickness (m), ordered from the seafloor (index 0,
    thickest layers) to the surface (last index, thinnest layers) — see
    :mod:`jemdiag.compute`. Each returned value is that layer's cell-center
    depth, computed by summing thickness from the surface downward, negated
    so that larger (less negative) values are shallower — 0 near the
    surface, increasingly negative toward the seafloor. This mirrors how
    the raw depth index already sorts (larger index = shallower), so
    swapping the index coordinate for this one leaves the default
    "surface at the top of the chart" layout (``invert_yaxis: false``)
    unchanged.
    """
    thickness = np.asarray(dzt.values, dtype=float)
    depth_to_bottom = np.cumsum(thickness[::-1])[::-1]
    depth_center = depth_to_bottom - thickness / 2
    return -depth_center


def has_real_horizontal_coords(da, x_dim, y_dim):
    """Return ``True`` unless both horizontal dims carry synthetic (index) coords."""
    return not (
        da[x_dim].attrs.get("jemdiag_synthetic", False)
        and da[y_dim].attrs.get("jemdiag_synthetic", False)
    )
