"""Area-weighted reductions and zonally-integrated ocean transports."""

import numpy as np
import xarray as xr

from . import grid

EARTH_RADIUS_M = 6_371_000.0  # mean Earth radius (m)

# Reference seawater properties used to convert v*T transport into watts for
# meridional heat transport. jem does not report these as run metadata, so
# fixed standard oceanographic values are used (consistent with common
# ocean-model defaults, e.g. MOM/NEMO); real seawater density/heat capacity
# vary with temperature and salinity, so this is an approximation.
RHO_SEAWATER = 1025.0  # kg/m^3
CP_SEAWATER = 3990.0  # J/(kg K)


def latitude_weights(lat_values):
    """Return ``cos(latitude)`` weights (latitude in degrees)."""
    return np.cos(np.deg2rad(np.asarray(lat_values, dtype=float)))


def weighted_mean(da, dims):
    """Mean of ``da`` over ``dims``.

    When the meridional dimension is reduced and has real (non-synthetic)
    coordinate values, the reduction is weighted by ``cos(latitude)`` to
    approximate an area-weighted average. Otherwise a plain mean is used.
    """
    _, y_dim = grid.horizontal_dims(da)
    if y_dim in dims and not da[y_dim].attrs.get("jemdiag_synthetic", False):
        weights = xr.DataArray(
            latitude_weights(da[y_dim].values),
            dims=[y_dim],
            coords={y_dim: da[y_dim]},
        )
        return da.weighted(weights).mean(dim=dims, keep_attrs=True)
    return da.mean(dim=dims, keep_attrs=True)


def _uniform_spacing(coord_values, dim_name):
    """Return the (constant) spacing of a 1D coordinate, or raise if irregular.

    Zonal-integral diagnostics below assume a regular longitude grid so that
    a single ``dx(lat)`` can be used for every column; jem's ocn grid is
    regular in longitude (verified against ``data/long_run``), but this is
    checked explicitly rather than assumed silently.
    """
    values = np.asarray(coord_values, dtype=float)
    if values.size < 2:
        raise ValueError(f"Need at least two '{dim_name}' points to infer grid spacing")
    diffs = np.diff(values)
    spacing = diffs[0]
    if not np.allclose(diffs, spacing, rtol=1e-6):
        raise ValueError(
            f"'{dim_name}' coordinate is not uniformly spaced; zonal-integral "
            "diagnostics (overturning streamfunction, heat transport) require "
            "a regular grid"
        )
    return spacing


def zonal_cell_width(lon_values, lat_values):
    """Physical zonal cell width in meters at each latitude: ``R*cos(lat)*dlon``.

    Assumes a regular (uniformly-spaced) longitude grid in degrees.
    """
    dlon_rad = np.deg2rad(_uniform_spacing(lon_values, "lon"))
    lat_rad = np.deg2rad(np.asarray(lat_values, dtype=float))
    return EARTH_RADIUS_M * np.cos(lat_rad) * dlon_rad


def overturning_streamfunction(v, dzt, mask, x_dim, y_dim, z_dim):
    """Meridional overturning streamfunction psi(depth, lat), in Sverdrups.

    ``v`` is meridional velocity (m/s) with dims including *x_dim*, *y_dim*,
    *z_dim* (already reduced over time). ``dzt`` is layer thickness (m)
    along *z_dim* only; ``mask`` is the land-sea mask (1=ocean, 0=land)
    sharing *v*'s dims, applied so land cells never contribute regardless of
    their raw velocity value.

    jem's ocn depth index runs from the seafloor (index 0, thickest layers)
    to the surface (last index, thinnest layers) — verified against
    ``data/long_run`` by matching the index-14 temperature profile to sea
    surface temperature. A cumulative sum from index 0 upward therefore
    yields psi = 0 at the seafloor and the full water-column transport at
    the surface, at each cumulative index representing the transport
    integrated from the seafloor up through that layer.
    """
    dx = zonal_cell_width(v[x_dim].values, v[y_dim].values)
    dx_da = xr.DataArray(dx, dims=[y_dim], coords={y_dim: v[y_dim]})
    transport = (v * mask * dx_da * dzt).sum(dim=x_dim)  # m^3/s per layer, dims (y_dim, z_dim)
    psi = transport.cumsum(dim=z_dim) / 1.0e6  # Sverdrups
    return psi.transpose(z_dim, y_dim)


def meridional_heat_transport(v, temp, dzt, mask, x_dim, y_dim, z_dim):
    """Meridional ocean heat transport MHT(lat), in petawatts.

    MHT = rho_seawater * cp_seawater * (zonal & vertical integral of v*T).
    ``temp`` is potential temperature in degC (jem convention); using degC
    rather than Kelvin is standard practice for this diagnostic since the
    net volume transport across a latitude circle is ~0, so the constant
    Kelvin offset does not affect the result.
    """
    dx = zonal_cell_width(v[x_dim].values, v[y_dim].values)
    dx_da = xr.DataArray(dx, dims=[y_dim], coords={y_dim: v[y_dim]})
    vt_flux = (v * temp * mask * dx_da * dzt).sum(dim=[x_dim, z_dim])  # degC * m^3/s
    mht_watts = RHO_SEAWATER * CP_SEAWATER * vt_flux
    return mht_watts / 1.0e15  # petawatts
